using Test, POMDPs, POMDPTools, Random
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Fixed adaptive actions and boundary outcomes" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    modes = (nominal, (true,false,false,false,false,false), ntuple(_->true,6))
    # Independent absolute displacements, ordered by heading and translation ID.
    offsets = (((0,1),(0,-1),(-1,0),(1,0)),
               ((1,0),(-1,0),(0,1),(0,-1)),
               ((0,-1),(0,1),(1,0),(-1,0)),
               ((-1,0),(1,0),(0,-1),(0,1)))
    @test collect(actions(model)) == collect(1:6)
    for x in 1:5, y in 1:5, (hi,heading) in enumerate((0,90,180,270))
        r = DS.DSState((x,y),heading)
        @test actions(model,r) == (r.quad == model.region_B ? Int64[] : collect(1:6))
        for mode in modes
            s = DS.adaptive_state(model,r,mode)
            @test actions(model,s) == (DS.is_terminal(s) ? Int64[] : collect(1:6))
            DS.is_terminal(s) && continue
            for a in 1:6
                outward = a <= 4 && !(1 <= x+offsets[hi][a][1] <= 5 &&
                                     1 <= y+offsets[hi][a][2] <= 5)
                successors = DS.feasible_successors(model,r,mode,a)
                @test length(successors) == 2*(7-count(identity,mode))
                @test any(pair -> pair[1] == DS.UNSAFE_CONFIG,successors) == outward
                @test DS.is_feasible_successor(model,r,mode,a,r,mode)
                distribution = DS.adaptive_transition(model,s,a)
                @test all(sp -> DS.is_unsafe(sp) || (1 <= sp.config.quad[1] <= 5 &&
                                                     1 <= sp.config.quad[2] <= 5),distribution.vals)
                if outward
                    @test DS.is_feasible_successor(model,r,mode,a,DS.UNSAFE_CONFIG,mode)
                    rewards = [reward(model,s,a,sp) for sp in distribution.vals]
                    @test Set(rewards) == Set([-1.0,-100.0])
                    @test sum(distribution.probs .* rewards) ≈ -50.5
                    key = DS.CountKey(model,r,mode,a,DS.UNSAFE_CONFIG,mode)
                    @test DS.count_value(model,DS.increment_count(model,s.counts,key),key) == 3
                    @test isempty(s.counts.increments)
                end
            end
        end
    end

    @testset "Outward commands have their own valid HDR rows" begin
        interior = DS.adaptive_state(model,DS.DSState((3,3)))
        unrelated = DS.locomotion_mode_uncertainty(model,interior,4;epsilon=.1,n_samples=64)
        theta = fill(1/14,14)
        for (a,position) in enumerate(((3,5),(3,1),(1,3),(5,3)))
            s = DS.adaptive_state(model,DS.DSState(position))
            hdr = DS.locomotion_mode_uncertainty(model,s,a;epsilon=.1,n_samples=64)
            @test length(hdr.successors) == 14
            @test any(pair -> pair[1] == DS.UNSAFE_CONFIG,hdr.successors)
            distribution = DS.robust_transition(model,s,a,theta;hdr)
            @test length(distribution.vals) == 14
            @test sum(prob for (sp,prob) in weighted_iterator(distribution) if DS.is_unsafe(sp)) ≈ .5
            @test_throws ArgumentError DS.robust_transition(model,s,a,theta;hdr=unrelated)
            @test isempty(s.counts.increments)
        end
    end

    @testset "Pose ambiguity cannot veto a command" begin
        for size in ((5,5),(1,3)), heading in (0,90,180,270), mode in DS.all_fault_modes()
            p = DS.DroneSurveillancePOMDP(size=size,obstacles=DS.DSPos[])
            @test actions(p,DS.adaptive_state(p,DS.DSState((1,1),heading),mode)) == collect(1:6)
        end
        poses = [DS.adaptive_state(model,DS.DSState(pos,heading))
                 for (pos,heading) in (((2,5),90),((1,5),90),((1,1),90),((1,5),0))]
        b = DS.ExactAdaptiveBelief(model,[poses[i] => (i==1 ? 0.0 : -1000.0)
                                         for i in eachindex(poses)];logweights=true)
        @test actions(model,b) == collect(1:6)
        vf = DS.RobustValueFunction(model;horizon=1,epsilon=.1,n_samples=64,seed=42)
        decision = DS.robust_decision(model,b,0,vf)
        @test [r.action for r in decision.action_results] == collect(1:6)
        @test all(isfinite(r.value) for r in decision.action_results)
        @test all(length(r.selected_rows._rows)==4 for r in decision.action_results)
        terminal = DS.adaptive_state(model,DS.UNSAFE_CONFIG)
        @test isempty(actions(model,DS.ExactAdaptiveBelief(model,terminal)))
        @test isempty(actions(model,DS.adaptive_state(model,DS.DSState(model.region_B))))
        @test actions(model,DS.ExactAdaptiveBelief(model,[terminal=>.5,poses[1]=>.5])) == collect(1:6)
        @test_throws ArgumentError DS.feasible_successors(model,poses[1].config,nominal,0)
        @test_throws ArgumentError DS.feasible_successors(model,poses[1].config,nominal,7)
    end
end
