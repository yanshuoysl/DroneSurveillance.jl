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
