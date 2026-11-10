using Test
using POMDPs
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end
if !isdefined(@__MODULE__, :run_adaptive_example)
    include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
end

@testset "Episode computation timing" begin
    DS = DroneSurveillanceAdaptive
    @testset "Compilation accounting and phase totals" begin
        stats = (;time=1.0,compile_time=0.2,recompile_time=0.1,gctime=0.05)
        phase = DS.computation_phase_timing(stats)
        # Recompilation is part of compile_time, not an additional subtraction.
        @test phase.execution_seconds ≈ 0.8
        @test phase.gc_seconds == 0.05
        step = (;planning=phase,environment=phase,belief_update=phase)
        total = DS.computation_timing_summary([step,step],9.0)
        @test total.steps == 2
        @test total.computation_seconds ≈ 4.8
        @test total.core_wall_seconds == 6.0
        @test total.compilation_seconds ≈ 1.2
        @test total.episode_wall_seconds == 9.0
        @test total.gc_seconds ≈ 0.3
        @test DS.computation_timing_summary(NamedTuple[],0.0).computation_seconds == 0
        # Older Julia @timed results cannot claim compilation-adjusted timing.
        unknown = DS.computation_phase_timing((;time=1.0,gctime=0.0))
        @test unknown.execution_seconds === nothing
        @test DS.computation_timing_summary([(;planning=unknown,environment=unknown,
            belief_update=unknown)],3.0).computation_seconds === nothing
    end

    @testset "Warm-up isolation, measured artifacts and deterministic replay" begin
        mktempdir() do dir
            args = ["42","2",dir,"2","--lookahead","1","--particles","128"]
            terminal = IOBuffer()
            measured = run_adaptive_example(vcat(args,"warmup=true");io=terminal,generate_gif=false)
            replay = run_adaptive_example(vcat(args,"warmup=false");io=nothing,generate_gif=false)
            @test measured.steps == measured.timing.steps == 2
            @test measured.timing.warmup_completed
            @test measured.timing.warmup.steps == 2
            @test !replay.timing.warmup_completed
            @test replay.timing.warmup === nothing
            @test !occursin("Compilation warm-up:",read(replay.console_log_path,String))
            @test dirname(replay.output_directory) == dir
            @test measured.total_reward == replay.total_reward
            @test [r.a for r in measured.history] == [r.a for r in replay.history]
            @test [r.o for r in measured.history] == [r.o for r in replay.history]
            @test measured.final_belief._logprobs == replay.final_belief._logprobs
            @test all(sum(values(s.counts.increments)) == 2 for s in support(measured.final_belief))
            @test measured.output_directory != replay.output_directory
            @test measured.gif_path === nothing
            @test isfile(measured.timing_path)
            @test measured.timing.core_wall_seconds >= 0
            @test measured.timing.episode_wall_seconds >= measured.timing.core_wall_seconds
            if VERSION >= v"1.11"
                @test measured.timing.computation_seconds >= 0
                @test measured.timing.compilation_seconds >= 0
                @test measured.timing.computation_seconds ≈ sum(
                    sum(r.timing[p].execution_seconds for p in (:planning,:environment,:belief_update))
                    for r in measured.history)
            end
            lines = readlines(measured.log_path)
            @test count(l -> occursin("\"event\":\"step_timing\"",l),lines) == 2
            @test count(l -> occursin("\"event\":\"start\"",l),lines) == 1
            @test occursin("\"timing\":",last(lines))
            @test occursin("\"warmup_completed\":true",read(measured.timing_path,String))
            @test occursin("\"timing_warmup_completed\":true",first(lines))
            @test occursin("N-step computation time",String(take!(terminal)))
            default = run_adaptive_example(["42","0",dir];io=nothing,generate_gif=false)
            @test !default.timing.warmup_completed
            disabled = run_adaptive_example(["42","0",dir,"warmup=true"];
                io=nothing,generate_gif=false,warmup=false)
            @test !disabled.timing.warmup_completed
            enabled = run_adaptive_example(["42","0",dir,"warmup=false"];
                io=nothing,generate_gif=false,warmup=true)
            @test enabled.timing.warmup_completed && enabled.timing.warmup.steps == 0
        end
    end
end
