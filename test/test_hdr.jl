using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Dirichlet HDR" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    s = DS.initial_adaptive_state(model)
    nominal = DS.NOMINAL_MODE
    failed_forward = (true, false, false, false, false, false)
    rp = DS.DSState([1, 2])
    successors = [(rp, nominal), (s.config, failed_forward)]

    @testset "PDF quantile and analytic threshold" begin
        # Reconstruct the calibration independently as a matrix of samples.
        counts = [2.0, 3.0, 4.0]
        distribution = DS.Dirichlet(counts)
        samples = rand(MersenneTwister(20), distribution, 100)
        densities = [DS.pdf(distribution, sample) for sample in eachcol(samples)]
        epsilon = 0.173
        expected = DS.quantile(densities, epsilon)
        tau = DS.estimate_tau(counts; epsilon, n_samples=100, rng=MersenneTwister(20))
        @test tau == expected
        @test tau != sort(densities)[floor(Int, 100epsilon)]
        @test tau == DS.estimate_tau(counts; epsilon, n_samples=100, rng=MersenneTwister(20))

        # For Dir(2,1), theta[1] ~ Beta(2,1), f(theta)=2theta[1].
        # Its lower epsilon density quantile is exactly 2sqrt(epsilon).
        analytic = 2sqrt(0.1)
        approximate = DS.estimate_tau([2, 1]; epsilon=0.1, n_samples=30_000,
                                     rng=MersenneTwister(42))
        @test isapprox(approximate, analytic; atol=0.03, rtol=0)
        @test DS.estimate_tau(ones(3); epsilon=0.05, n_samples=50) ≈ 2.0
        @test DS.estimate_tau([7]; epsilon=0.05, n_samples=50) == 1.0
    end

    @testset "Hyperstate row and continuous membership" begin
        uniform = DS.build_row_hdr(model, s, 1, successors;
                                  epsilon=0.05, n_samples=100, rng=MersenneTwister(1))
        @test uniform.distribution.alpha == [1, 1]
        @test uniform.successors == successors
        @test uniform.epsilon == 0.05
        @test uniform.n_samples == 100
        @test uniform.tau == 1.0
        @test DS.contains(uniform, [0.0, 1.0])
        @test DS.contains(uniform, [1.0, 0.0])
        @test [0.123456789, 0.876543211] in uniform
        @test !DS.contains(uniform, [-0.1, 1.1])
        @test !DS.contains(uniform, [0.1, 0.2])
        @test !DS.contains(uniform, [1.0])
        @test !DS.contains(uniform, [NaN, 0.0])
        @test !DS.contains(uniform, [Inf, 0.0])
        @test !DS.contains(uniform, [1.0 + 1e-10, 0.0])

        key = DS.CountKey(s.config, nominal, 1, rp, nominal)
        phi = DS.increment_count(s.counts, key)
        learned = DS.AdaptiveState(s.config, nominal, phi)
        hdr = DS.build_row_hdr(model, learned, 1, successors;
                              epsilon=0.1, n_samples=5_000, rng=MersenneTwister(4))
        @test hdr.distribution.alpha == [2, 1]
        @test isempty(s.counts.increments)
        @test DS.count_value(learned.counts, key) == 2
        @test [1.0, 0.0] in hdr
        @test !([0.0, 1.0] in hdr)
        @test [0.7, 0.3] in hdr

        reversed = DS.build_row_hdr(model, learned, 1, reverse(successors);
                                   epsilon=0.1, n_samples=100, rng=MersenneTwister(4))
        @test reversed.distribution.alpha == [1, 2]
        @test reversed.successors == reverse(successors)
        @test [0.0, 1.0] in reversed
        @test !([1.0, 0.0] in reversed)

        # The object snapshots the successor list and the counts at construction.
        saved = copy(successors)
        successors[1] = (DS.DSState([2, 1]), nominal)
        @test hdr.successors == saved
        successors[1] = saved[1]
        learned.counts.increments[key] = 20
        @test hdr.distribution.alpha == [2, 1]

        # Fresh samples check approximate posterior coverage, not a guarantee.
        held_out = rand(MersenneTwister(99), hdr.distribution, 10_000)
        coverage = count(theta -> DS.contains(hdr, theta), eachcol(held_out)) / 10_000
        @test abs(coverage - 0.9) < 0.03

        terminal_support = [(DS.UNSAFE_CONFIG, nominal),
                            (DS.DSState(model.region_B, 90), nominal)]
        terminal_hdr = DS.build_row_hdr(model, s, 1, terminal_support;
                                       epsilon=0.1, n_samples=50)
        @test terminal_hdr.successors == terminal_support
        @test [0.5, 0.5] in terminal_hdr
        singleton = DS.build_row_hdr(model, s, 1, [(rp, nominal)]; epsilon=0.1, n_samples=1)
        @test singleton.tau == 1.0
        @test [1.0] in singleton
        @test !([0.0] in singleton)
    end

    @testset "Input and support validation" begin
        for bad_counts in (Float64[], [0.0, 1.0], [-1.0, 1.0], [NaN, 1.0], [Inf, 1.0])
            @test_throws ArgumentError DS.estimate_tau(bad_counts; epsilon=0.1, n_samples=10)
        end
        for epsilon in (0.0, 1.0, -0.1, 1.1, NaN, Inf)
            @test_throws ArgumentError DS.estimate_tau([1, 1]; epsilon, n_samples=10)
        end
        for n_samples in (0, -1)
            @test_throws ArgumentError DS.estimate_tau([1, 1]; epsilon=0.1, n_samples)
        end
        # Uniform density (179!) overflows Float64: direct PDF mode must report it.
        @test_throws DomainError DS.estimate_tau(ones(180); epsilon=0.1, n_samples=2,
                                                rng=MersenneTwister(1))
        @test_throws ArgumentError DS.build_row_hdr(model, s, 0, successors; epsilon=0.1)
        @test_throws ArgumentError DS.build_row_hdr(model, s, 1, []; epsilon=0.1)
        @test_throws ArgumentError DS.build_row_hdr(model, s, 1,
            [successors[1], successors[1]]; epsilon=0.1)
        @test_throws ArgumentError DS.build_row_hdr(model, s, 1,
            [(DS.DSState([100, 100]), nominal)]; epsilon=0.1)
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2)])
        @test_throws ArgumentError DS.build_row_hdr(blocked, s, 1,
            [(DS.DSState([2, 2]), nominal)]; epsilon=0.1)
        two_faults = (true, true, false, false, false, false)
        @test_throws ArgumentError DS.build_row_hdr(model, s, 1,
            [(rp, two_faults)]; epsilon=0.1)
        faulty = DS.AdaptiveState(s.config, failed_forward)
        @test_throws ArgumentError DS.build_row_hdr(model, faulty, 1,
            [(rp, nominal)]; epsilon=0.1)
        for terminal in (DS.adaptive_state(model, DS.UNSAFE_CONFIG),
                         DS.adaptive_state(model, DS.DSState(model.region_B)),
                         DS.AdaptiveState(DS.DSState(model.region_B)))
            @test_throws ArgumentError DS.build_row_hdr(model, terminal, 1, successors; epsilon=0.1)
        end
    end
end
