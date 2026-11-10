using Test
using Random
using POMDPs
using POMDPTools
using Distributions: logpdf

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "KKT Dirichlet HDR row minimization" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    config = DS.DSState((3, 3), 0)
    nominal = DS.NOMINAL_MODE
    all_faults = ntuple(_ -> true, 6)
    pairs = DS.feasible_successors(model, config, all_faults, 4)

    # Prescribed thresholds make these deterministic mathematical fixtures;
    # they do not assert posterior coverage for a particular epsilon.
    fixture(phi, tau) = DS.DirichletHDR(DS.hdr_distribution(phi), copy(pairs[1:length(phi)]), 0.1, 1, tau)

    function check_result(hdr, values, result; log_atol=1e-9)
        @test all(>(0), result.theta)
        @test sum(result.theta) ≈ 1
        @test result.theta in hdr
        @test DS.SelectedTransitionRow(result.theta, hdr) isa DS.SelectedTransitionRow
        @test result.value ≈ sum(values .* result.theta)
        @test result.log_hdr_slack ≈ logpdf(hdr.distribution, result.theta) - log(hdr.tau) atol=1e-12
        @test 0 <= result.log_hdr_slack <= log_atol
        @test 0 <= result.optimality_gap
        @test result.objective_lower_bound ≈ result.value - result.optimality_gap
        @test result.relative_stationarity_residual <= 1e-10
        @test result.shifted_kappa > 0 && result.lambda > 0
        @test result.kappa > -minimum(values)
    end

    @testset "Analytic two-outcome solutions and the slide equation" begin
        # Dir(2,2) has density 6*x*(1-x). For tau=1, x lies between
        # (1-sqrt(1/3))/2 and (1+sqrt(1/3))/2.
        hdr = fixture([2, 2], 1.0)
        low, high = (1 - sqrt(1 / 3)) / 2, (1 + sqrt(1 / 3)) / 2
        for values in ([-1.0, 2.0], [2.0, -1.0], [0.0, 1.0],
                       [1.0, 1.0 + 1e-12], [1e12, nextfloat(1e12)],
                       [-1e150, 2e150], [1e-150, -2e-150], [-1e308, 1e308])
            result = DS.minimize_hdr_row(hdr, values)
            check_result(hdr, values, result)
            expected = values[1] < values[2] ? [high, low] : [low, high]
            @test result.theta ≈ expected atol=1e-8 rtol=0
            @test result.kappa > -minimum(values)
            exponents = hdr.distribution.alpha .- 1
            # The library log-density and the user's slides have the same root.
            c = log(hdr.tau) + hdr.distribution.lmnB
            slide_residual = sum(exponents .* log.(result.theta)) - c
            @test slide_residual ≈ result.log_hdr_slack atol=1e-12
            weights = BigFloat.(exponents) ./ ((BigFloat.(values) .- minimum(values)) .+ result.shifted_kappa)
            @test weights ./ sum(weights) ≈ result.theta
            exact_value = sum(values .* expected)
            @test result.value >= exact_value - 1e-14 * maximum(abs, values)
            @test result.objective_lower_bound <= exact_value + 1e-14 * maximum(abs, values)
        end
        # Close to the mode and close to a simplex endpoint exercise both
        # bracket expansion directions without crossing a pole.
        for tau in (1.5 * (1 - 1e-6), 1e-200)
            hdr = fixture([2, 2], tau)
            result = DS.minimize_hdr_row(hdr, [-1.0, 2.0])
            check_result(hdr, [-1.0, 2.0], result)
            low = (tau / 3) / (1 + sqrt(1 - 2tau / 3))
            @test result.theta[2] ≈ low rtol=1e-7
        end
    end

    @testset "Asymmetric counts and independent scalar probability search" begin
        hdr = fixture([4, 3], 1.0)
        mode = 3 / 5
        # Search in probability space, independently of kappa or its formula.
        lower, upper = mode, 1.0
        for _ in 1:70
            x = (lower + upper) / 2
            if DS.pdf(hdr.distribution, [x, 1-x]) >= hdr.tau
                lower = x
            else
                upper = x
            end
        end
        result = DS.minimize_hdr_row(hdr, [-3.0, 7.0])
        check_result(hdr, [-3.0, 7.0], result)
        @test result.theta[1] ≈ lower atol=1e-8
    end

    @testset "Fourteen successors, learned counts, ties, permutation, and snapshots" begin
        successors = DS.feasible_successors(model, config, nominal, 4)
        keys = [DS.CountKey(model, config, nominal, 4, rp, mp) for (rp, mp) in successors]
        counts = DS.DirichletCounts(Dict(key => j-1 for (j, key) in enumerate(keys)))
        state = DS.adaptive_state(model, config, nominal, counts)
        hdr = DS.build_row_hdr(model, state, 4; epsilon=0.1, n_samples=2048, rng=MersenneTwister(71))
        values = Float64.([3, -4, 1, 8, 2, -4, 7, 3, 0, 5, 11, -2, 1, 4])
        saved_values, saved_phi = copy(values), copy(hdr.distribution.alpha)
        saved_successors, saved_counts, saved_tau = copy(hdr.successors), copy(counts.increments), hdr.tau
        result = DS.minimize_hdr_row(hdr, values)
        check_result(hdr, values, result)
        @test length(result.theta) == 14
        mode = (saved_phi .- 1) ./ sum(saved_phi .- 1)
        @test result.value <= sum(values .* mode)
        samples = rand(MersenneTwister(72), hdr.distribution, 2000)
        feasible_samples = [sample for sample in eachcol(samples) if sample in hdr]
        @test !isempty(feasible_samples)
        @test all(result.value <= sum(values .* sample) + 1e-7 for sample in feasible_samples)
        @test values == saved_values && hdr.distribution.alpha == saved_phi
        @test hdr.successors == saved_successors && hdr.tau == saved_tau
        @test counts.increments == saved_counts

        permutation = randperm(MersenneTwister(73), 14)
        permuted_hdr = DS.DirichletHDR(DS.hdr_distribution(saved_phi[permutation]),
            saved_successors[permutation], hdr.epsilon, hdr.n_samples, hdr.tau)
        reordered = DS.minimize_hdr_row(permuted_hdr, values[permutation])
        @test reordered.theta ≈ result.theta[permutation] atol=1e-8
        @test reordered.value ≈ result.value
        transformed = DS.minimize_hdr_row(hdr, 3 .* values .+ 200)
        @test transformed.theta ≈ result.theta atol=1e-8
        @test transformed.value ≈ 3result.value + 200
        @test transformed.kappa ≈ 3result.kappa - 200
        @test transformed.lambda ≈ 3result.lambda

        # Later caller edits do not modify the already returned solution.
        values .= 99
        hdr.distribution.alpha[1] = 200
        hdr.successors[1] = hdr.successors[end]
        @test sum(saved_values .* result.theta) ≈ result.value
        @test result.theta in DS.DirichletHDR(DS.hdr_distribution(saved_phi),
            saved_successors, 0.1, 2048, saved_tau)
    end

    @testset "Constant values, singleton, degenerate set, and concentrated counts" begin
        hdr = fixture([4, 3], 1.0)
        for values in ([0.0, 0.0], [-10.0, -10.0])
            result = DS.minimize_hdr_row(hdr, values)
            @test result.status == :constant_values
            @test result.theta ≈ [0.6, 0.4]
            @test result.value ≈ values[1]
            @test result.iterations == result.bracket_iterations == 0
            @test result.kappa === nothing && result.lambda == 0
            @test result.optimality_gap == 0 && result.theta in hdr
        end
        for tau in (0.5, 1.0)
            singleton = DS.minimize_hdr_row(fixture([2], tau), [-7.0])
            @test singleton.status == :singleton
            @test singleton.theta == [1] && singleton.value == -7
            @test singleton.kappa === nothing && singleton.lambda === nothing
        end
        phi = [4.0, 3.0]
        distribution = DS.hdr_distribution(phi)
        mode = (phi .- 1) ./ sum(phi .- 1)
        maximum_pdf = DS.pdf(distribution, mode)
        only_mode = DS.minimize_hdr_row(fixture(phi, maximum_pdf), [-1.0, 2.0])
        @test only_mode.status == :mode_only && only_mode.theta ≈ mode
        @test only_mode.kappa === nothing && only_mode.lambda === nothing
        @test only_mode.optimality_gap == 0
        concentrated = fixture([2000, 3000], DS.estimate_tau([2000, 3000];
            epsilon=0.1, n_samples=2048, rng=MersenneTwister(74)))
        result = DS.minimize_hdr_row(concentrated, [-100.0, 100.0])
        check_result(concentrated, [-100.0, 100.0], result)
    end

    @testset "Invalid inputs and explicit convergence failure" begin
        hdr = fixture([2, 2], 1.0)
        for values in (Float64[], [1.0], [NaN, 1.0], [Inf, 1.0])
            @test_throws ArgumentError DS.minimize_hdr_row(hdr, values)
        end
        for phi in ([1, 1], [2, 1], [0.5, 2])
            @test_throws ArgumentError DS.minimize_hdr_row(fixture(phi, 0.1), [-1.0, 2.0])
        end
        for tau in (0.0, -1.0, NaN, Inf, nextfloat(1.5))
            @test_throws ArgumentError DS.minimize_hdr_row(fixture([2, 2], tau), [-1.0, 2.0])
        end
        @test_throws ArgumentError DS.minimize_hdr_row(fixture([2], 1.1), [0.0])
        duplicate = DS.DirichletHDR(hdr.distribution, [pairs[1], pairs[1]], 0.1, 1, 1.0)
        @test_throws ArgumentError DS.minimize_hdr_row(duplicate, [-1.0, 2.0])
        malformed = DS.DirichletHDR(hdr.distribution, pairs[1:1], 0.1, 1, 1.0)
        @test_throws ArgumentError DS.minimize_hdr_row(malformed, [-1.0, 2.0])
        for (epsilon, n_samples) in ((0.0, 1), (0.1, 0))
            bad = DS.DirichletHDR(hdr.distribution, copy(pairs), epsilon, n_samples, 1.0)
            @test_throws ArgumentError DS.minimize_hdr_row(bad, [-1.0, 2.0])
        end
        for kwargs in ((;log_atol=0.0), (;objective_atol=-1.0), (;objective_rtol=NaN),
                       (;kkt_rtol=Inf), (;max_iterations=0), (;max_bracket_iterations=0),
                       (;objective_atol=0.0, objective_rtol=0.0))
            @test_throws ArgumentError DS.minimize_hdr_row(hdr, [-1.0, 2.0]; kwargs...)
        end
        stale = fixture([2, 2], 1.0)
        stale.distribution.alpha[1] = 4
        @test_throws ArgumentError DS.minimize_hdr_row(stale, [-1.0, 2.0])
        @test_throws DS.RowMinimizationError DS.minimize_hdr_row(hdr, [-1.0, 2.0]; max_iterations=1)
        tight = fixture([2, 2], 1.5 * (1 - 1e-6))
        @test_throws DS.RowMinimizationError DS.minimize_hdr_row(tight, [-1.0, 2.0]; max_bracket_iterations=1)
    end

    @testset "One-step reward coefficients from the actual drone model" begin
        p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(3, 5)])
        source = DS.adaptive_state(p, DS.DSState((4, 5), 0))
        minima = Dict{Int,DS.HDRRowMinimum}()
        for a in (2, 3, 4)
            hdr = DS.build_row_hdr(p, source, a; epsilon=0.1, n_samples=2048, rng=MersenneTwister(76))
            coefficients = [DS.adaptive_reward(p, source, a, DS.adaptive_state(p, rp, mp))
                            for (rp, mp) in hdr.successors]
            minima[a] = DS.minimize_hdr_row(hdr, coefficients)
        end
        @test minima[2].status == :constant_values && minima[2].value ≈ -1
        @test minima[4].value > -1
        @test sum(minima[4].theta[8:14]) < 0.5 # nature reduces goal entry
        @test minima[3].value < -1
        @test sum(minima[3].theta[8:14]) > 0.5 # nature increases collision
        @test isempty(source.counts.increments)
    end

    @testset "Transition and exact belief update integration" begin
        source = DS.initial_adaptive_state(model)
        b = DS.ExactAdaptiveBelief(model, source)
        a = 4
        hdr = DS.build_row_hdr(model, source, a; epsilon=0.1, n_samples=2048, rng=MersenneTwister(75))
        values = vcat(fill(-1.0, 7), fill(9.0, 7)) # fixed illustrative continuation coefficients
        result = DS.minimize_hdr_row(hdr, values)
        check_result(hdr, values, result)
        @test all(result.theta[1] ≈ q for q in result.theta[1:7])
        @test all(result.theta[8] ≈ q for q in result.theta[8:14])
        @test sum(result.theta[1:7]) > 0.5
        kernel = DS.robust_transition(model, source, a, result.theta; hdr)
        @test kernel.probs ≈ result.theta
        @test isempty(source.counts.increments)
        @test all(sum(Base.values(sp.counts.increments)) == 1 for sp in kernel.vals)
        selected = DS.SelectedTransitionRow(result.theta, hdr)
        prepared = DS.BeliefTransitionRows(model, b, a, Dict(source => selected))
        updater = DS.ExactAdaptiveUpdater(model)
        DS.prepare_belief_update!(updater, b, a, prepared)
        obs = DS.DSObservation(3, 1, 1)
        posterior = update(updater, b, a, obs)
        evidence = sum(w * pdf(observation(model, a, sp), obs) for (sp, w) in weighted_iterator(kernel))
        @test length(posterior) == 14
        @test sum(w for (_, w) in posterior) ≈ 1
        @test all(pdf(posterior, sp) ≈ w * pdf(observation(model, a, sp), obs) / evidence
                  for (sp, w) in weighted_iterator(kernel))
        @test all(sum(Base.values(sp.counts.increments)) == 1 for sp in support(posterior))
        @test isempty(source.counts.increments)
    end
end
