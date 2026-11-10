using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Uniform episode fault scheduling" begin
    DS = DroneSurveillanceAdaptive
    @test DS.sample_fault_schedule(0, 2) == Int[]
    @test DS.sample_fault_schedule(100, 0) == Int[]
    @test_throws ArgumentError DS.sample_fault_schedule(-1, 2)
    @test_throws ArgumentError DS.sample_fault_schedule(100, -1)
    @test_throws ArgumentError DS.sample_fault_schedule(big(typemax(Int)) + 1, 2)

    @testset "Sorted distinct times, bounded event count, and reproducibility" begin
        for (H, N_f) in ((1, 2), (2, 6), (5, 2), (100, 2), (100, 6), (100, 10))
            rng = MersenneTwister(42)
            for _ in 1:100
                schedule = DS.sample_fault_schedule(H, N_f; rng=rng)
                @test length(schedule) <= min(N_f, H, 6)
                @test issorted(schedule)
                @test all(t -> 1 <= t <= H, schedule)
                @test length(unique(schedule)) == length(schedule)
            end
        end
        first_rng, second_rng = MersenneTwister(17), MersenneTwister(17)
        first_run = [DS.sample_fault_schedule(100, 2; rng=first_rng) for _ in 1:50]
        second_run = [DS.sample_fault_schedule(100, 2; rng=second_rng) for _ in 1:50]
        @test first_run == second_run
        @test length(unique(Tuple.(first_run))) > 3
    end

    @testset "Uniform K and uniform subsets conditional on K" begin
        rng = MersenneTwister(60)
        n = 18_000
        counts = Dict{Tuple, Int}()
        for _ in 1:n
            key = Tuple(DS.sample_fault_schedule(4, 2; rng=rng))
            counts[key] = get(counts, key, 0) + 1
        end
        # All 1 + 4 + 6 possible schedules are observed. Size classes each
        # carry probability 1/3, rather than giving every schedule equal mass.
        @test length(counts) == 11
        for k in 0:2
            observed = sum(v for (key, v) in counts if length(key) == k)
            @test abs(observed - n / 3) < 6sqrt(n * (1 / 3) * (2 / 3))
        end
        for (key, observed) in counts
            p = 1 / (3 * binomial(4, length(key)))
            @test abs(observed - n * p) < 6sqrt(n * p * (1 - p))
        end
    end
end

if !isdefined(@__MODULE__, :run_adaptive_example)
    include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
end

@testset "Optional fixed fault time in example runs" begin
    DS = DroneSurveillanceAdaptive
    defaults = adaptive_example_settings(["42","30"])
    @test defaults.fault_time_step === nothing
    for option in (["--fault_time_step","4"],["--fault_time_step=4"])
        settings = adaptive_example_settings(vcat(["42","30"],option))
        @test settings.fault_time_step == 4
        @test settings.H == 30 && settings.output_root == defaults.output_root
    end
    @test adaptive_example_settings(["--fault_time_step","1","42","1"]).fault_time_step == 1
    @test adaptive_example_settings(["42","30","--fault_time_step","30"]).fault_time_step == 30
    for option in (["--fault_time_step"],["--fault_time_step="],["--fault_time_step","oops"],
                   ["--fault_time_step","1.5"],["--fault_time_step","0"],
                   ["--fault_time_step","-1"],["--fault_time_step","31"],
                   ["--fault_time_step","4","--fault_time_step=5"])
        @test_throws ArgumentError adaptive_example_settings(vcat(["42","30"],option))
    end
    @test_throws ArgumentError adaptive_example_settings(["42","0","--fault_time_step","1"])
    @test_throws ArgumentError adaptive_example_settings(["42","30","/tmp/fixed_fault","0","--fault_time_step","4"])
    mktempdir() do dir
        args = ["42","4",dir,"2","--lookahead","1","--particles","128"]
        fixed_args = vcat(args,["--fault_time_step","4"])
        fixed = run_adaptive_example(fixed_args;io=nothing,generate_gif=false)
        @test fixed.fault_time_step == 4 && fixed.fault_steps == (4,)
        @test fixed.steps == 4
        @test all(r.fault_event === nothing for r in fixed.history[1:3])
        event = fixed.history[4].fault_event
        @test event.step == 4 && event.action in 1:6
        @test count(identity,fixed.final_state.mode) == 1
        @test fixed.final_state.mode[event.action]
        @test occursin("fault_time_step=4",read(fixed.console_log_path,String))
        @test occursin("\"fault_steps\":[4]",read(fixed.log_path,String))
        warmed = run_adaptive_example(vcat(fixed_args,"warmup=true");io=nothing,generate_gif=false)
        @test warmed.timing.warmup_completed && warmed.timing.warmup.steps == 4
        @test [r.fault_event for r in warmed.history] == [r.fault_event for r in fixed.history]
        @test [r.a for r in warmed.history] == [r.a for r in fixed.history]
        @test [r.o for r in warmed.history] == [r.o for r in fixed.history]
        @test warmed.final_belief._logprobs == fixed.final_belief._logprobs
        random = run_adaptive_example(args;io=nothing,generate_gif=false)
        schedule_rng = MersenneTwister(rand(MersenneTwister(42),UInt))
        @test random.fault_time_step === nothing
        @test random.fault_steps == Tuple(DS.sample_fault_schedule(4,2;rng=schedule_rng))
    end
end

@testset "Explicit fault type in example runs" begin
    DS = DroneSurveillanceAdaptive
    @test adaptive_example_settings(["42","30"]).fault_type === nothing
    for action in 1:6, option in (["--fault_type",string(action)],["--fault_type=$action"])
        settings = adaptive_example_settings(vcat(["42","30","--fault_time_step","4"],option))
        @test settings.fault_type == action && settings.fault_time_step == 4
    end
    @test adaptive_example_settings(["--fault_type","1","42","30","--fault_time_step","4"]).fault_type == 1
    for option in (["--fault_type"],["--fault_type="],["--fault_type","oops"],
                   ["--fault_type","1.5"],["--fault_type","0"],["--fault_type","7"],
                   ["--fault_type","-1"],["--fault_type","1","--fault_type=2"])
        @test_throws ArgumentError adaptive_example_settings(vcat(["42","30","--fault_time_step","4"],option))
    end
    @test_throws ArgumentError adaptive_example_settings(["42","30","--fault_type","1"])
    mktempdir() do dir
        args = ["42","6",dir,"2","--lookahead","1","--particles","128",
            "--fault_time_step","4","--fault_type","1"]
        result = run_adaptive_example(args;io=nothing,generate_gif=false)
        @test result.fault_type == 1 && result.fault_time_step == 4 && result.fault_steps == (4,)
        @test result.steps == 6
        @test all(r.fault_event === nothing && !r.sp.mode[1] for r in result.history[1:3])
        fourth = result.history[4]
        @test fourth.fault_event == (;step=4,action=1)
        @test fourth.a == 1 && fourth.sp.config == fourth.s.config
        @test !fourth.s.mode[1] && fourth.sp.mode[1]
        @test all(r.s.mode[1] && r.sp.mode[1] && r.fault_event === nothing for r in result.history[5:6])
        @test count(identity,result.final_state.mode) == 1
        @test occursin("fault_type=1",read(result.console_log_path,String))
        @test occursin("\"fault_actions\":[1]",read(result.log_path,String))
        warmed = run_adaptive_example(vcat(args,"warmup=true");io=nothing,generate_gif=false)
        @test warmed.timing.warmup_completed && warmed.timing.warmup.steps == 6
        @test [r.fault_event for r in warmed.history] == [r.fault_event for r in result.history]
        @test [r.a for r in warmed.history] == [r.a for r in result.history]
        @test [r.o for r in warmed.history] == [r.o for r in result.history]
        @test warmed.final_belief._logprobs == result.final_belief._logprobs
    end
end
