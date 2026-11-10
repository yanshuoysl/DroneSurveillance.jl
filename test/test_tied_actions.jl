using Test
using Random
using POMDPs
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end
if !isdefined(@__MODULE__, :run_adaptive_example)
    include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
end

@testset "Optional random action ties" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    b = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((3,3))))
    vf = DS.RobustValueFunction(p;horizon=1,epsilon=0.1,n_samples=32,seed=42)
    rng_fingerprint(rng) = rand(copy(rng),UInt,8)

    @testset "Default, reproducible ties, partial ties, and unique maxima" begin
        default = DS.robust_decision(p,b,0,vf)
        @test default.action == 1
        @test all(isapprox(r.value,-1.0;atol=1e-12) for r in default.action_results)
        rng = MersenneTwister(42)
        before = rng_fingerprint(rng)
        @test DS.robust_decision(p,b,0,vf;rng).action == 1
        @test rng_fingerprint(rng) == before
        random_sequence(seed) = begin
            tie_rng = MersenneTwister(seed)
            [DS.robust_decision(p,b,0,vf;tied_action_rdn=true,rng=tie_rng).action for _ in 1:36]
        end
        sequence = random_sequence(42)
        @test sequence == random_sequence(42)
        @test sequence != random_sequence(43)
        @test Set(sequence) == Set(1:6)
        corner = DS.ExactAdaptiveBelief(p)
        corner_decision = DS.robust_decision(p,corner,0,vf)
        @test all(r.value < -1 for r in corner_decision.action_results if r.action in (2,3))
        @test all(DS.robust_decision(p,corner,0,vf;tied_action_rdn=true,rng).action in (1,4,5,6)
                  for _ in 1:24)
        near_goal = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((4,5))))
        before = rng_fingerprint(rng)
        @test DS.robust_decision(p,near_goal,0,vf;tied_action_rdn=true,rng).action == 4
        @test rng_fingerprint(rng) == before
    end

    @testset "Pending choices retain their transition rows; reset restarts the seed" begin
        policy = DS.RobustPolicy(vf;tied_action_rdn=true,tie_seed=43)
        a = action(policy,b)
        pending = policy._pending
        before = rng_fingerprint(policy._tie_rng)
        @test action(policy,b) == a && policy._pending === pending
        @test rng_fingerprint(policy._tie_rng) == before
        chosen = only(filter(r -> r.action == a,pending.action_results))
        @test pending.selected_rows === chosen.selected_rows
        @test policy._exact_updater._pending === chosen.selected_rows
        DS.reset_robust_policy!(policy)
        @test action(policy,b) == a
        DS.reset_robust_policy!(policy)
        initialize_belief(updater(policy),b)
        @test action(policy,b) == a
        @test !DS.RobustPolicy(vf).tied_action_rdn
        @test_throws ArgumentError DS.RobustPolicy(vf;tie_seed=-1)
        @test_throws ArgumentError DS.RecedingHorizonRobustPolicy(vf;H=2,tie_seed=-1)
    end

    @testset "Replanning advances a separate stream and reset replays it" begin
        source = DS.adaptive_state(p,DS.DSState((3,3)),ntuple(_ -> true,6))
        initial = DS.ExactAdaptiveBelief(p,source)
        policy = DS.RecedingHorizonRobustPolicy(vf;H=2,tied_action_rdn=true,tie_seed=42)
        run_episode() = DS.simulate_robust(p,policy;H=2,N_f=6,seed=42,
            fault_steps=[],initial_state=source,initial_belief=initial,io=nothing)
        result = run_episode()
        expected_rng = MersenneTwister(42)
        expected = [rand(expected_rng,collect(1:6)) for _ in 1:2]
        @test [r.a for r in result.history] == expected
        @test all(isapprox(c.value,-1;atol=1e-12) for r in result.history for c in r.decision.action_values)
        @test rng_fingerprint(policy._tie_rng) == rng_fingerprint(expected_rng)
        @test rng_fingerprint(policy._particle_rng) == rng_fingerprint(MersenneTwister(policy.particle_seed))
        DS.reset_robust_policy!(policy)
        replay = run_episode()
        @test [r.a for r in replay.history] == expected
        @test [r.o for r in replay.history] == [r.o for r in result.history]
    end

    @testset "Run option reaches policy, logs, and warm-up; keyword overrides CLI" begin
        mktempdir() do dir
            args = ["42","1",dir,"0","--lookahead","1","tied_action_rdn=True"]
            random_run = run_adaptive_example(args;io=nothing,generate_gif=false)
            @test random_run.tied_action_rdn && random_run.policy.tied_action_rdn
            @test random_run.policy._policy.tied_action_rdn
            @test occursin("tied_action_rdn=true",read(random_run.console_log_path,String))
            @test occursin("\"tied_action_rdn\":true",first(readlines(random_run.log_path)))
            warmed = run_adaptive_example(vcat(args,"warmup=true");io=nothing,generate_gif=false)
            @test [r.a for r in warmed.history] == [r.a for r in random_run.history]
            @test warmed.final_belief._logprobs == random_run.final_belief._logprobs
            fixed = run_adaptive_example(args;io=nothing,generate_gif=false,tied_action_rdn=false)
            default = run_adaptive_example(args[1:end-1];io=nothing,generate_gif=false)
            @test !fixed.tied_action_rdn && !default.tied_action_rdn
            @test [r.a for r in fixed.history] == [r.a for r in default.history]
            @test fixed.final_belief._logprobs == default.final_belief._logprobs
            @test occursin("\"tied_action_rdn\":false",first(readlines(default.log_path)))
            @test random_run.fault_steps == fixed.fault_steps
        end
    end
end
