using Test
using Random
using POMDPs
using POMDPTools

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

struct TerminationTestUpdater <: Updater
    calls::Vector{NamedTuple}
end

function POMDPs.update(up::TerminationTestUpdater, b, a, o)
    push!(up.calls, (; b, a, o))
    return (; updates=length(up.calls), observation=o)
end

@testset "Adaptive simulation termination" begin
    DS = DroneSurveillanceAdaptive
    nominal = DS.NOMINAL_MODE
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    start = DS.adaptive_state(model, DS.DSState((3, 3)))
    goal = DS.adaptive_state(model, DS.DSState(model.region_B))
    unsafe = DS.adaptive_state(model, DS.UNSAFE_CONFIG)

    function commanded_kernel(p, s, a)
        hdr = DS.locomotion_mode_uncertainty(p, s, a; epsilon=0.1,
                                            n_samples=12, rng=MersenneTwister(7))
        theta = zeros(length(hdr.successors))
        j = findfirst(pair -> pair[1] != s.config && pair[2] == s.mode, hdr.successors)
        theta[j] = 1
        return DS.robust_transition(p, s, a, theta; hdr=hdr)
    end

    function stay_kernel(p, s, a)
        successors = DS.feasible_successors(p, s.config, s.mode, a)
        theta = zeros(length(successors))
        theta[findfirst(==((s.config, s.mode)), successors)] = 1
        return DS.adaptive_transition(p, s, a, theta)
    end

    @testset "Physical terminal hook and horizon priority" begin
        @test !isterminal(model, start)
        @test isterminal(model, goal)
        @test isterminal(model, unsafe)
        @test DS.termination_reason(model, start, 0, 5) === nothing
        @test DS.termination_reason(model, start, 5, 5) == :horizon
        @test DS.termination_reason(model, start, 6, 5) == :horizon
        @test DS.termination_reason(model, goal, 5, 5) == :goal
        @test DS.termination_reason(model, unsafe, 5, 5) == :collision
        @test DS.termination_reason(model, goal, 0, 0) == :goal
        @test DS.termination_reason(model, unsafe, 0, 0) == :collision
        @test_throws ArgumentError DS.termination_reason(model, start, -1, 5)
        @test_throws ArgumentError DS.termination_reason(model, start, 0, -1)
        malformed = DS.AdaptiveState(start.config, nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.termination_reason(model, malformed, 0, 5)
        headed = DS.DroneSurveillancePOMDP(initial_heading=90, obstacles=DS.DSPos[])
        @test DS.initial_adaptive_state(headed).config.heading == 90
    end

    @testset "Goal/collision stop immediately, retaining entry reward and counts" begin
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)])
        for (p, source, expected_reason, expected_reward) in (
                (model, DS.adaptive_state(model, DS.DSState((4, 5))), :goal, 100.0),
                (blocked, DS.adaptive_state(blocked, DS.DSState((3, 3))), :collision, -100.0)),
                H in (1, 10)
            calls = NamedTuple[]
            up = TerminationTestUpdater(calls)
            policy_calls = Ref(0)
            initial_belief = :agent_belief
            policy = FunctionPolicy(b -> begin
                @test b === initial_belief
                policy_calls[] += 1
                policy_calls[] == 1 || error("Action selected after termination")
                return 4
            end)
            kernel_calls = Ref(0)
            kernel = (p, s, a) -> begin
                kernel_calls[] += 1
                return commanded_kernel(p, s, a)
            end
            result = DS.simulate_adaptive(p, policy, up, initial_belief;
                        H=H, initial_state=source, transition_kernel=kernel,
                        rng=MersenneTwister(17))
            @test result.reason == expected_reason
            @test result.terminated && !result.truncated
            @test result.steps == length(result.history) == 1
            @test policy_calls[] == kernel_calls[] == length(calls) == 1
            entry = only(result.history)
            @test entry.r == expected_reward
            @test entry.s === source
            @test entry.sp === result.final_state
            @test result.history isa POMDPTools.SimHistory
            @test discounted_reward(result.history) == expected_reward
            @test calls[1] == (; b=initial_belief, a=4, o=DS.TERMINAL_OBSERVATION)
            @test result.final_belief == (; updates=1, observation=DS.TERMINAL_OBSERVATION)
            key = DS.CountKey(p, source.config, nominal, 4, entry.sp.config, nominal)
            @test result.final_state.counts.increments == Dict(key => 1)
            @test isempty(source.counts.increments)
            @test isempty(actions(p, result.final_state))
            @test render(p, entry) !== nothing
        end
    end

    @testset "Stay and rotation actions count toward exactly H" begin
        for kernel in (stay_kernel, commanded_kernel), H in (1, 4, 7)
            policy_calls = Ref(0)
            beliefs = Any[]
            policy = FunctionPolicy(b -> begin
                policy_calls[] += 1
                push!(beliefs, b)
                return 5
            end)
            up = TerminationTestUpdater(NamedTuple[])
            result = DS.simulate_adaptive(model, policy, up, :initial;
                        H=H, initial_state=start, transition_kernel=kernel,
                        rng=MersenneTwister(19))
            @test result.reason == :horizon
            @test !result.terminated && result.truncated
            @test !isterminal(model, result.final_state)
            @test result.steps == length(result.history) == policy_calls[] == H
            @test length(up.calls) == H
            @test result.final_state.config.quad == start.config.quad
            @test result.final_state.config.heading == (kernel === stay_kernel ? 0 : mod(90H, 360))
            @test sum(step.r for step in result.history) == -H
            @test all(step.r == -1 && step.a == 5 for step in result.history)
            @test collect(step.t for step in result.history) == collect(1:H)
            @test sum(values(result.final_state.counts.increments)) == H
            @test beliefs[1] === :initial
            @test all(b -> !(b isa DS.AdaptiveState), beliefs)
            @test result.final_belief.updates == H
            @test isempty(start.counts.increments)
            for (i, entry) in enumerate(result.history)
                @test sum(values(entry.sp.counts.increments)) == i
                @test sum(values(entry.s.counts.increments); init=0) == i - 1
            end
        end
    end

    @testset "H=0 and initially terminal states execute no callbacks" begin
        never = (args...) -> error("Callback invoked without an executable step")
        policy = FunctionPolicy(never)
        up = TerminationTestUpdater(NamedTuple[])
        for (source, H, expected_reason) in ((start, 0, :horizon),
                (goal, 0, :goal), (unsafe, 0, :collision),
                (goal, 10, :goal), (unsafe, 10, :collision))
            result = DS.simulate_adaptive(model, policy, up, :initial;
                        H=H, initial_state=source, transition_kernel=never)
            @test result.reason == expected_reason
            @test result.steps == 0
            @test isempty(result.history)
            @test result.final_state === source
            @test result.final_belief === :initial
            @test isempty(up.calls)
        end
        @test_throws ArgumentError DS.simulate_adaptive(model, policy, up, nothing;
                    H=-1, transition_kernel=never)
    end

    @testset "Boundary-invalid actions are rejected before the kernel" begin
        policy = PlaybackPolicy([4])
        edge = DS.adaptive_state(model, DS.DSState((5, 3)))
        up = TerminationTestUpdater(NamedTuple[])
        kernel_calls = Ref(0)
        kernel = (args...) -> (kernel_calls[] += 1; error("Outward kernel invoked"))
        @test_throws ArgumentError DS.simulate_adaptive(model, policy, up, nothing;
                    H=1, initial_state=edge, transition_kernel=kernel)
        @test kernel_calls[] == 0
        @test isempty(up.calls)
        @test isempty(edge.counts.increments)
    end

    @testset "Augmented camera observations retain the known sensor model" begin
        for a in 1:6
            d = observation(model, a, start)
            original = observation(model, a, start.config)
            @test all(pdf(d, o) == pdf(original, o) for o in observations(model))
            for s in (goal, unsafe)
                @test pdf(observation(model, a, s), DS.TERMINAL_OBSERVATION) == 1
            end
        end
    end

    @testset "Drone route example stops at the goal or configured H" begin
        if !isdefined(@__MODULE__, :random_demo_environment)
            include("adaptive_demo_route.jl")
        end
        p, route = random_demo_environment(MersenneTwister(42), DS.DSPos(3, 3))
        for H in (3, length(route), length(route) + 6)
            result = DS.simulate_adaptive(p, PlaybackPolicy(route), NothingUpdater(), nothing;
                        H=H, transition_kernel=commanded_kernel, rng=MersenneTwister(23))
            @test result.steps == min(H, length(route))
            @test result.reason == (H < length(route) ? :horizon : :goal)
            @test sum(values(result.final_state.counts.increments)) == result.steps
            if result.reason == :goal
                positions = [entry.sp.config.quad for entry in result.history]
                @test DS.DSPos(3, 3) in positions
                @test findfirst(==(p.region_B), positions) == length(result.history)
                @test last(result.history).r == 100
            end
        end
    end
end
