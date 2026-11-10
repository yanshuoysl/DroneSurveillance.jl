using Test
using Random
using POMDPs
using POMDPTools
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end

@testset "Hidden fault integration fixtures" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    start = DS.adaptive_state(p,DS.DSState((3,3)))

    @testset "Every action freezes immediately and permanently; joint count key" begin
        for a in 1:6
            other = mod1(a+1,6)
            env = DS.HiddenFaultEnvironment(3,2;fault_steps=[1,3],fault_actions=[a,other])
            first = DS.physical_step(p,env,start,a,1)
            @test first.sp.config == start.config
            @test first.event == (;step=1,action=a)
            @test first.sp.mode[a]
            key = first.count_key
            @test key.mode == DS.NOMINAL_MODE
            @test key.next_mode == first.sp.mode
            @test DS.count_value(first.sp.counts,key) == 3
            @test isempty(start.counts.increments)
            @test sum(values(first.sp.counts.increments)) == 1
            @test a in actions(p,first.sp)
            # A different, still healthy command executes normally.
            second = DS.physical_step(p,env,first.sp,other,2)
            @test second.event === nothing
            @test second.sp.config == DS.commanded_config(first.sp.config,other)
            @test second.sp.mode == first.sp.mode
            third = DS.physical_step(p,env,second.sp,a,3)
            @test third.sp.config == second.sp.config
            @test count(identity,third.sp.mode) == 2
            @test third.count_key.mode == second.sp.mode
            @test third.count_key.next_mode == third.sp.mode
            @test sum(values(third.sp.counts.increments)) == 3
            @test length(DS.feasible_successors(p,start.config,DS.NOMINAL_MODE,a)) == 14
            @test length(DS.feasible_successors(p,first.sp.config,first.sp.mode,a)) == 12
        end
    end

    @testset "Healthy motion, collision, boundaries, fault timing and validation" begin
        healthy = DS.HiddenFaultEnvironment(3,0;fault_steps=[])
        for a in 1:6
            out = DS.physical_step(p,healthy,start,a,1)
            @test out.sp.config == DS.commanded_config(start.config,a)
            @test out.sp.mode == start.mode
            @test out.event === nothing
        end
        delayed = DS.HiddenFaultEnvironment(2,1;fault_steps=[2],fault_actions=[4])
        first = DS.physical_step(p,delayed,start,4,1)
        second = DS.physical_step(p,delayed,first.sp,4,2)
        @test first.sp.config.quad == DS.DSPos(4,3)
        @test second.sp.config == first.sp.config
        @test second.count_key.mode == DS.NOMINAL_MODE
        edge = DS.adaptive_state(p,DS.DSState((5,3)),ntuple(i->i==4,6))
        frozen = DS.HiddenFaultEnvironment(1,1;fault_steps=[],initial_mode=edge.mode)
        frozen_outward = DS.physical_step(p,frozen,edge,4,1)
        @test frozen_outward.sp.config == edge.config
        @test reward(p,edge,4,frozen_outward.sp) == -1
        healthy_edge = DS.adaptive_state(p,edge.config)
        boundary_impact = DS.physical_step(p,healthy,healthy_edge,4,1)
        @test DS.is_unsafe(boundary_impact.sp)
        @test reward(p,healthy_edge,4,boundary_impact.sp) == -100
        @test sum(values(boundary_impact.sp.counts.increments)) == 1
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4,3)])
        impact = DS.physical_step(blocked,healthy,start,4,1)
        @test DS.is_unsafe(impact.sp)
        @test impact.attempted == DS.DSState((4,3))
        @test reward(blocked,start,4,impact.sp) == -100
        @test_throws ArgumentError DS.physical_step(blocked,healthy,impact.sp,5,2)
        stays = DS.physical_step(blocked,DS.HiddenFaultEnvironment(1,1;fault_steps=[1],fault_actions=[4]),start,4,1)
        @test !DS.is_unsafe(stays.sp)
        @test reward(blocked,start,4,stays.sp) == -1
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,1;fault_steps=[1,2])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,2;fault_steps=[2,1])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,2;fault_steps=[1,1])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,2;fault_steps=[0])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,2;fault_steps=[1,2],fault_actions=[4,4])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,1;fault_steps=[1],fault_actions=[7])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,1;fault_actions=[4])
        @test_throws ArgumentError DS.HiddenFaultEnvironment(2,0;initial_mode=edge.mode)
        # A generous budget must still respect the number of healthy actions
        # when starting from an already faulty physical mode.
        mostly_failed = (true,true,true,true,true,false)
        for seed in 1:20
            env = DS.HiddenFaultEnvironment(10,10;initial_mode=mostly_failed,schedule_rng=MersenneTwister(seed))
            @test length(env.fault_steps) <= 1
        end
    end

    @testset "Reproducible random schedules and healthy-only action selection" begin
        for seed in 1:40
            e1 = DS.HiddenFaultEnvironment(10,2;schedule_rng=MersenneTwister(seed),fault_rng=MersenneTwister(seed+1))
            e2 = DS.HiddenFaultEnvironment(10,2;schedule_rng=MersenneTwister(seed),fault_rng=MersenneTwister(seed+1))
            @test e1.fault_steps == e2.fault_steps
            @test length(e1.fault_steps) <= 2
            s1,s2 = start,start
            events = Int[]
            for t in 1:10
                x = DS.physical_step(p,e1,s1,5,t)
                y = DS.physical_step(p,e2,s2,5,t)
                @test x == y
                x.event === nothing || push!(events,x.event.action)
                s1,s2 = x.sp,y.sp
            end
            @test length(unique(events)) == length(events) == length(e1.fault_steps)
            @test count(identity,s1.mode) == length(events)
            @test sum(values(s1.counts.increments)) == 10
        end
    end

    @testset "One-step robust wiring fixtures, logs, chart and terminal entry" begin
        # Focused runner checks, not the final fitted example/benchmark.
        model = DS.DroneSurveillancePOMDP(region_A=DS.DSPos(4,5),obstacles=[DS.DSPos(3,5)])
        b = DS.ExactAdaptiveBelief(model)
        vf = DS.RobustValueFunction(model;horizon=1,epsilon=0.1,n_samples=128,seed=9)
        for fails in (false,true)
            mktempdir() do dir
                policy = DS.RobustPolicy(vf)
                terminal = IOBuffer()
                log = joinpath(dir,"episode.jsonl")
                csv = joinpath(dir,"rewards.csv")
                chart = joinpath(dir,"rewards.svg")
                result = DS.simulate_robust(model,policy;H=1,N_f=1,initial_belief=b,
                    fault_steps=fails ? [1] : Int[],fault_actions=fails ? [4] : Int[],
                    io=terminal,log_path=log,reward_csv_path=csv,reward_chart_path=chart)
                @test result.steps == 1
                @test only(result.history).a == 4
                @test result.reason == (fails ? :horizon : :goal)
                @test result.total_reward == (fails ? -1 : 100)
                @test policy.stage == 1 && policy._pending === nothing
                @test policy._exact_updater._pending === nothing
                @test sum(values(result.final_state.counts.increments)) == 1
                @test all(sum(values(s.counts.increments)) == 1 for s in support(result.final_belief))
                @test occursin("step,reward\n1,$(Float64(result.total_reward))",read(csv,String))
                json = read(log,String)
                @test occursin("\"event\":\"decision\"",json)
                @test occursin("\"selected_rows\":[",json)
                @test occursin("\"event\":\"belief_branch\"",json)
                @test occursin("\"count_before\":2,\"count_after\":3,\"delta\":1",json)
                @test occursin("\"event\":\"stop\"",json)
                @test occursin("simulator_truth",json)
                @test occursin("Immediate reward by step",read(chart,String))
                @test occursin("<svg",read(chart,String))
                stdout = String(take!(terminal))
                @test occursin("action values:",stdout)
                @test occursin("belief after:",stdout)
                @test occursin("realized phi key: 2 -> 3",stdout)
                # Independently recompute with the PRE-observation selected model.
                reference = DS.robust_decision(model,b,0,vf)
                observed = only(result.history).o
                expected = DS.exact_belief_update(model,b,4,observed,reference.selected_rows)
                @test expected._logprobs == result.final_belief._logprobs
                @test_throws ArgumentError DS.simulate_robust(model,DS.RobustPolicy(vf);H=2,io=nothing)
                @test_throws ArgumentError DS.simulate_robust(model,DS.RobustPolicy(vf);log_path=log,io=nothing)
            end
        end
        # A two-step controlled freeze uses updated branch counts on decision 2.
        vf2 = DS.RobustValueFunction(model;horizon=2,epsilon=0.1,n_samples=128,seed=9)
        policy = DS.RobustPolicy(vf2)
        result = DS.simulate_robust(model,policy;H=2,N_f=1,fault_steps=[1],fault_actions=[4],io=nothing)
        @test result.steps == 2 && policy.stage == 2
        @test result.reason == :horizon
        @test result.final_state.mode[4]
        @test sum(values(result.final_state.counts.increments)) == 2
        @test all(sum(values(s.counts.increments)) == 2 for s in support(result.final_belief))
        @test only(filter(s->s.t==2,collect(result.history))).decision.stage == 1
        @test any(any(>(2),hdr.distribution.alpha) for hdr in values(vf2._hdrs))
    end

    @testset "Empty episodes, artifacts and output validation" begin
        vf = DS.RobustValueFunction(p;horizon=0,epsilon=0.1,n_samples=32)
        mktempdir() do dir
            policy = DS.RobustPolicy(vf)
            chart = joinpath(dir,"empty.svg")
            result = DS.simulate_robust(p,policy;N_f=0,io=nothing,reward_chart_path=chart)
            @test result.steps == 0 && result.reason == :horizon
            @test policy.stage == 0
            @test occursin("No actions executed",read(chart,String))
            @test_throws ArgumentError DS.simulate_robust(p,policy;io=nothing,log_path=chart,reward_csv_path=chart)
            @test_throws ArgumentError DS.simulate_robust(p,policy;io=nothing,reward_chart_path=joinpath(dir,"bad.png"))
            buffer = IOBuffer()
            DS.episode_json(buffer,(;text="line\nquote\"slash\\tab\t",value=-Inf))
            @test String(take!(buffer)) == "{\"text\":\"line\\u000aquote\\\"slash\\\\tab\\u0009\",\"value\":\"-Inf\"}"
            DS.save_reward_chart(joinpath(dir,"trend.svg"),[(;t=1,r=-1.0),(;t=2,r=-1.0),(;t=3,r=100.0)])
            @test occursin("Immediate reward by step",read(joinpath(dir,"trend.svg"),String))
        end
        # Merely loading the entry point must not fit/run or write output.
        include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
        @test isdefined(@__MODULE__,:run_adaptive_example)
    end
end
