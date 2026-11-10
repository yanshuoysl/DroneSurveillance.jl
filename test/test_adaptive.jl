using Test
using Random
using POMDPs
using POMDPTools
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end

@testset "AR model and configuration geometry" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(size=(3,4),region_A=[2,2],obstacles=DS.DSPos[])
    s = rand(MersenneTwister(1),initialstate(p))
    @test statetype(p) == DS.AdaptiveState
    @test s == DS.initial_adaptive_state(p)
    @test s.config == DS.DSState((2,2))
    @test s.mode == DS.NOMINAL_MODE && isempty(s.counts.increments)
    @test fieldnames(DS.DSState) == (:quad,:heading)
    headed = DS.DroneSurveillancePOMDP(initial_heading=90,obstacles=DS.DSPos[])
    @test rand(initialstate(headed)).config.heading == 90
    @test_throws ArgumentError DS.DSState((2,2),45)
    # No misleading finite configuration enumeration of the hyperstate space.
    @test !applicable(states,p)
    @test !applicable(stateindex,p,s)
    @test !applicable(transition,p,s,1) # callers must select a learned/physical model
    @test !applicable(transition,p,s.config,1)
    @test ordered_actions(p) == collect(1:6)
    @test discount(p) == 0.95
    moves = ((0,((2,3),(2,1),(1,2),(3,2))),
             (90,((3,2),(1,2),(2,3),(2,1))),
             (180,((2,1),(2,3),(3,2),(1,2))),
             (270,((1,2),(3,2),(2,1),(2,3))))
    for (heading,positions) in moves
        r = DS.DSState((2,2),heading)
        for (a,pos) in enumerate(positions)
            @test DS.commanded_config(r,a) == DS.DSState(pos,heading)
            @test actionindex(p,a) == a
        end
        @test DS.commanded_config(r,5) == DS.DSState(r.quad,mod(heading+90,360))
        @test DS.commanded_config(r,6) == DS.DSState(r.quad,mod(heading-90,360))
        @test render(p,(s=r,)) !== nothing
    end
    @test render(p,(s=DS.adaptive_state(p,DS.DSState(p.region_B)),)) !== nothing
    @test render(p,(s=DS.adaptive_state(p,DS.UNSAFE_CONFIG),)) !== nothing
    @test render(p) !== nothing

    @testset "Known camera on configurations and hyperstates" begin
        model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        @test length(observations(model)) == 65
        @test obstype(model) == DS.DSObservation
        for (i,o) in enumerate(observations(model))
            @test obsindex(model,o) == i
        end
        @test_throws ArgumentError obsindex(model,DS.DSObservation(0,1,1))
        for (heading,cells) in ((0,((2,4),(3,4),(4,4))),
                               (90,((4,4),(4,3),(4,2))),
                               (180,((4,2),(3,2),(2,2))),
                               (270,((2,2),(2,3),(2,4))))
            r = DS.DSState((3,3),heading)
            @test DS.visible_cells(model,r) == map(DS.DSPos,cells)
            @test !DS.in_fov(model,r,r.quad)
        end
        r = DS.DSState((4,4))
        correct = DS.DSObservation(1,1,2)
        d = observation(model,1,r)
        @test pdf(d,correct) ≈ 0.9^3
        @test pdf(d,DS.DSObservation(2,1,2)) ≈ (0.1/3)*0.9^2
        @test sum(pdf(d,o) for o in observations(model)) ≈ 1
        @test DS.observation_labels(correct) == (:ordinary,:ordinary,:survey)
        perfect = DS.DroneSurveillancePOMDP(camera=DS.KnownCamera(correct_probability=1),obstacles=DS.DSPos[])
        @test rand(observation(perfect,1,r)) == correct
        @test rand(observation(perfect,1,DS.DSState((1,1)))) == DS.DSObservation(3,1,1)
        custom = DS.KnownCamera([0.8 0.15 0.03 0.02; 0.1 0.7 0.15 0.05; 0.2 0.1 0.6 0.1; 0.1 0.1 0.1 0.7])
        cp = DS.DroneSurveillancePOMDP(camera=custom,obstacles=DS.DSPos[])
        @test pdf(observation(cp,1,r),correct) ≈ 0.8^2*0.7
        for prob in (-0.1,1.1,NaN,Inf)
            @test_throws ArgumentError DS.KnownCamera(correct_probability=prob)
        end
        @test_throws ArgumentError DS.KnownCamera(ones(2,2))
        @test_throws ArgumentError DS.KnownCamera(zeros(4,4))
        # Exhaust configurations to check the camera without enumerating phi.
        for x in 1:5,y in 1:5,h in DS.HEADINGS
            config = DS.DSState((x,y),h)
            state = DS.adaptive_state(model,config)
            d = observation(model,1,state)
            @test sum(pdf(d,o) for o in observations(model)) ≈ 1
            if DS.is_terminal(state)
                @test rand(d) == DS.TERMINAL_OBSERVATION
            else
                @test all(pdf(d,o) == pdf(observation(model,1,config),o) for o in observations(model))
            end
        end
        @test DS.visible_cells(model,DS.UNSAFE_CONFIG) == ()
        @test rand(observation(model,1,DS.adaptive_state(model,DS.UNSAFE_CONFIG))) == DS.TERMINAL_OBSERVATION
    end

    @testset "Map validation and deterministic map sampling" begin
        first_map = DS.random_obstacles((5,5);exclude=[DS.DSPos(1,1),DS.DSPos(5,5)],rng=MersenneTwister(7))
        @test first_map == DS.random_obstacles((5,5);exclude=[DS.DSPos(1,1),DS.DSPos(5,5)],rng=MersenneTwister(7))
        @test length(unique(first_map)) == 2
        @test isempty(DS.random_obstacles((1,1);count=0))
        @test_throws ArgumentError DS.random_obstacles((1,1);count=2)
        @test_throws ArgumentError DS.random_obstacles((0,5))
        for obstacles in ([DS.DSPos(1,1)],[DS.DSPos(5,5)],[DS.DSPos(6,1)],[DS.DSPos(2,2),DS.DSPos(2,2)])
            @test_throws AssertionError DS.DroneSurveillancePOMDP(;obstacles)
        end
        @test_throws MethodError reward(p,s.config,1)
        @test_throws MethodError initialize_belief(DiscreteUpdater(p),initialstate(p))
    end
end
