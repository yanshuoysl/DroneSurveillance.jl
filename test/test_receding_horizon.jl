using Test
using Random
using POMDPs
using POMDPTools
if !isdefined(@__MODULE__,:DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end
if !isdefined(@__MODULE__,:adaptive_example_settings)
    include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
end

@testset "Independent episode horizon and receding lookahead" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(1,2)],
        camera=DS.KnownCamera(correct_probability=1))
    all_failed = ntuple(_ -> true,6)
    source = DS.adaptive_state(model,DS.DSState((1,1)),all_failed)
    b0 = DS.ExactAdaptiveBelief(model,source)
    template(L;kwargs...) = DS.RobustValueFunction(model;horizon=L,epsilon=0.1,n_samples=64,seed=7,kwargs...)

    @testset "Command line keeps H independent of L" begin
        settings = adaptive_example_settings(["42","30"])
        @test settings.H == 30 && settings.lookahead == 2
        @test settings.N_f == 2
        @test !settings.warmup
        @test !settings.tied_action_rdn
        for option in (["tied_action_rdn=true"],["--tied_action_rdn=True"],
                       ["--tied_action_rdn","true"])
            parsed = adaptive_example_settings(vcat(["42","30"],option))
            @test parsed.tied_action_rdn && parsed.output_root == settings.output_root
        end
        @test !adaptive_example_settings(["tied_action_rdn=False"]).tied_action_rdn
        @test adaptive_example_settings(["42","30","/tmp/tied_action_rdn=true"]).output_root ==
            "/tmp/tied_action_rdn=true"
        @test_throws ArgumentError adaptive_example_settings(["tied_action_rdn=yes"])
        @test_throws ArgumentError adaptive_example_settings(["tied_action_rdn="])
        @test_throws ArgumentError adaptive_example_settings(["--tied_action_rdn"])
        @test_throws ArgumentError adaptive_example_settings(["tied_action_rdn=true","--tied_action_rdn=false"])
        cli = adaptive_example_settings(["42","30","--lookahead","2","--particles","128","warmup=false"])
        @test !cli.warmup && cli.output_root == settings.output_root
        @test cli.H == 30 && cli.lookahead == 2 && cli.particles == 128
        @test adaptive_example_settings(["warmup=true","42","30"]).warmup
        @test adaptive_example_settings(["42","30","--warmup=true"]).warmup
        @test !adaptive_example_settings(["42","30","--warmup","false"]).warmup
        @test adaptive_example_settings(["42","30","/tmp/warmup=false"]).output_root == "/tmp/warmup=false"
        @test_throws ArgumentError adaptive_example_settings(["42","30","warmup=yes"])
        @test_throws ArgumentError adaptive_example_settings(["42","30","warmup="])
        @test_throws ArgumentError adaptive_example_settings(["42","30","--warmup"])
        @test_throws ArgumentError adaptive_example_settings(["42","30","warmup=false","--warmup=true"])
        explicit = adaptive_example_settings(["42","30","/tmp/ar_parent","1","--lookahead","3"])
        @test explicit.H == 30 && explicit.lookahead == 3
        @test explicit.output_root == "/tmp/ar_parent" && explicit.N_f == 1
        @test adaptive_example_settings(["--lookahead=1","42","30"]).lookahead == 1
        @test_throws ArgumentError adaptive_example_settings(["42","30","--lookahead","0"])
        @test_throws ArgumentError adaptive_example_settings(["--lookahead"])
        @test_throws ArgumentError adaptive_example_settings(["--lookahead","1","--lookahead=2"])
        @test_throws ArgumentError adaptive_example_settings(["--unknown"])
        @test_throws ArgumentError adaptive_example_settings(["42","-1"])
        p = DS.RecedingHorizonRobustPolicy(template(2);H=30)
        @test p.episode_horizon == 30 && p.value_function.horizon == 2
        @test isempty(p.value_function._families) && p._policy === nothing
        @test occursin("H=30, L=2",sprint(show,p))
        @test_throws ArgumentError DS.RecedingHorizonRobustPolicy(template(0);H=30)
        @test_throws ArgumentError DS.RecedingHorizonRobustPolicy(template(2);H=-1)
        @test_throws ArgumentError DS.RecedingHorizonRobustPolicy(template(2);H=2,stage=3)
    end

    @testset "Fresh local plans preserve exact counts and prepared rows" begin
        policy = DS.RecedingHorizonRobustPolicy(template(2);H=5)
        up = updater(policy)
        b = b0
        true_state = source
        env = DS.HiddenFaultEnvironment(5,6;initial_mode=all_failed,fault_steps=[])
        previous_vf = nothing
        for step in 1:5
            prior = copy(b._logprobs)
            a = action(policy,b)
            @test a == 4
            @test policy.stage == step-1
            @test policy._policy.value_function.horizon == min(2,6-step)
            @test policy._policy.stage == 0
            @test policy._policy.value_function !== previous_vf
            @test policy._fit_report.points <= 1000
            pending = policy._policy._pending
            local_vf = policy._policy.value_function
            @test maximum(only(values(pending.selected_rows._rows)).alpha) == step+1
            @test pending.selected_rows._prior == prior
            @test action(policy,b) == a
            @test policy._policy._pending === pending
            @test policy._policy.value_function === local_vf
            if step == 1
                @test_throws ArgumentError update(up,b,5,DS.DSObservation(3,4,1))
                @test_throws DS.ZeroObservationEvidence update(up,b,a,DS.TERMINAL_OBSERVATION)
                @test policy.stage == 0 && policy._policy._pending === pending
                policy.stage = 1
                @test_throws ArgumentError action(policy,b)
                @test_throws ArgumentError update(up,b,a,DS.DSObservation(3,4,1))
                policy.stage = 0
                wrong_counts = DS.increment_count(source.counts,
                    DS.CountKey(model,source.config,source.mode,4,DS.DSState((2,1)),source.mode))
                wrong_b = DS.ExactAdaptiveBelief(model,DS.adaptive_state(model,source.config,source.mode,wrong_counts))
                @test_throws ArgumentError action(policy,wrong_b)
            end
            physical = DS.physical_step(model,env,true_state,a,step)
            o = rand(MersenneTwister(step),observation(model,a,physical.sp))
            expected = DS.exact_belief_update(model,b,a,o,pending.selected_rows)
            posterior = update(up,b,a,o)
            @test posterior._logprobs == expected._logprobs
            @test b._logprobs == prior
            @test policy.stage == step && policy._pending_stage === nothing
            @test policy._policy._pending === nothing
            @test length(posterior) == 1
            @test only(support(posterior)).config == source.config
            @test sum(values(only(support(posterior)).counts.increments)) == step
            previous_vf = local_vf
            true_state,b = physical.sp,posterior
        end
        @test_throws ArgumentError action(policy,b)
        DS.reset_robust_policy!(policy)
        @test policy.stage == 0 && policy._policy === nothing && policy._fit_report === nothing
        @test_throws ArgumentError update(up,b0,4,DS.DSObservation(3,4,1))
        @test pdf(initialize_belief(up,Deterministic(source)),source) == 1
        @test policy.stage == 0
        @test_throws ArgumentError DS.reset_robust_policy!(policy;stage=6)
        goal = DS.adaptive_state(model,DS.DSState(model.region_B))
        @test_throws ArgumentError action(policy,DS.ExactAdaptiveBelief(model,goal))
        @test policy._policy === nothing
    end

    @testset "Episode budget drives faults and termination beyond the window" begin
        vf = template(1)
        policy = DS.RecedingHorizonRobustPolicy(vf;H=3)
        mktempdir() do dir
            terminal = IOBuffer()
            log = joinpath(dir,"episode.jsonl")
            result = DS.simulate_robust(model,policy;H=3,N_f=1,
                fault_steps=[3],fault_actions=[1],io=terminal,log_path=log)
            @test result.steps == 3 && result.reason == :horizon
            @test policy.stage == 3
            @test [r.decision.episode_stage for r in result.history] == [0,1,2]
            @test all(r.decision.stage == 0 && r.decision.lookahead == 1 for r in result.history)
            @test all(r.decision.planning_mode == :receding_horizon for r in result.history)
            @test result.history[1].fault_event === nothing
            @test result.history[2].fault_event === nothing
            @test result.history[3].fault_event == (;step=3,action=1)
            @test result.final_state.mode[1]
            @test sum(values(result.final_state.counts.increments)) == 3
            @test all(sum(values(s.counts.increments)) == 3 for s in support(result.final_belief))
            text = read(log,String)
            @test occursin("\"episode_horizon\":3",text)
            @test occursin("\"planning_lookahead\":1",text)
            @test occursin("\"planning_mode\":\"receding_horizon\"",text)
            @test occursin("\"episode_stage\":2,\"lookahead\":1",text)
            @test occursin("planning_stage=0, L=1",String(take!(terminal)))
        end
        @test_throws ArgumentError DS.simulate_robust(model,DS.RecedingHorizonRobustPolicy(vf;H=3);H=2,io=nothing)
        zero = DS.RecedingHorizonRobustPolicy(template(2);H=0)
        empty_result = DS.simulate_robust(model,zero;N_f=0,io=nothing)
        @test empty_result.steps == 0 && zero.stage == 0 && zero._policy === nothing
        # Planning failure is explicit and occurs before any physical command.
        limited = template(2;options=DS.RobustPlannerOptions(max_training_points=1))
        fail = DS.RecedingHorizonRobustPolicy(limited;H=30)
        @test_throws DS.RobustPlanningError action(fail,b0)
        @test fail.stage == 0 && fail._pending_stage === nothing && fail._policy === nothing
        @test isempty(source.counts.increments)
        # Oversized structural expansions must fail before costly calibration.
        small = template(1;options=DS.RobustPlannerOptions(max_hypotheses=1))
        ambiguous = DS.ExactAdaptiveBelief(model,
            DS.adaptive_state(model,DS.DSState((3,3)),all_failed))
        states = DS.planning_support(small,ambiguous)
        @test_throws DS.RobustPlanningError DS.prepare_robust_action!(small,states,4)
        @test isempty(small._hdrs)
    end
end
