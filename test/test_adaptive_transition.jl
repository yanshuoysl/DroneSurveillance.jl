using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Adaptive transition kernels" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    r = DS.DSState((3, 3), 0)
    nominal = DS.NOMINAL_MODE
    s = DS.adaptive_state(model, r)

    @testset "Predictive probabilities and branch-specific counts" begin
        for mode in DS.all_fault_modes(), action in 1:6
            successors = DS.feasible_successors(model, r, mode, action)
            keys = [DS.CountKey(model, r, mode, action, rp, mp) for (rp, mp) in successors]
            # Distinct concentrations 1:n give a nonuniform, independently
            # specified oracle for every joint destination and every row size.
            phi = DS.DirichletCounts(Dict(key => j - 1 for (j, key) in enumerate(keys)))
            before = copy(phi.increments)
            state = DS.adaptive_state(model, r, mode, phi)
            n = length(successors)
            expected = collect(1:n) ./ sum(1:n)
            row = DS.posterior_predictive_row(model, state, action)
            @test row.successors == successors
            @test row.counts == collect(1:n)
            @test row.probabilities ≈ expected
            distribution = DS.adaptive_transition(model, state, action)
            @test length(distribution.vals) == n
            @test sum(distribution.probs) ≈ 1.0
            for (j, sp) in enumerate(distribution.vals)
                @test (sp.config, sp.mode) == successors[j]
                @test DS.pdf(distribution, sp) ≈ expected[j]
                @test sp.counts.increments[keys[j]] == j
                @test all(get(sp.counts.increments, key, 0) == increment + Int(key == keys[j])
                          for (key, increment) in before)
                @test sum(values(sp.counts.increments)) == sum(values(before)) + 1
                # A physical destination with unincremented history is not the
                # correct next hyperstate, even though pose and mode match.
                @test DS.pdf(distribution, DS.adaptive_state(model, sp.config, sp.mode, phi)) == 0
            end
            @test phi.increments == before
        end
        for (mode, n) in ((nominal, 14), ((true, false, false, false, false, false), 12))
            state = DS.adaptive_state(model, r, mode)
            row = DS.posterior_predictive_row(model, state, 4)
            @test row.counts == ones(Int, n)
            @test row.probabilities ≈ fill(1 / n, n)
        end
        # Learning on right/nominal must not change a different action's row.
        key = DS.CountKey(model, r, nominal, 4, DS.DSState((4, 3), 0), nominal)
        phi = DS.increment_count(model, s.counts, key)
        learned = DS.adaptive_state(model, r, nominal, phi)
        @test DS.posterior_predictive_row(model, learned, 1).probabilities ≈ fill(1 / 14, 14)
        row = DS.posterior_predictive_row(model, learned, 4)
        @test row.probabilities[8] ≈ 2 / 15
        @test row.probabilities[1] ≈ 1 / 15
        # Large valid entries must not overflow an Int row sum.
        large = DS.DirichletCounts(Dict(DS.CountKey(model, r, nominal, 4, rp, mp) => typemax(Int) - 1
                                       for (rp, mp) in row.successors))
        @test DS.posterior_predictive_row(model, DS.adaptive_state(model, r, nominal, large), 4).
              probabilities ≈ fill(1 / 14, 14)
    end

    @testset "Explicit joint rows, structural zeros, and sampling" begin
        theta = zeros(14)
        theta[1], theta[end] = 0.2, 0.8
        distribution = DS.adaptive_transition(model, s, 4, theta)
        @test distribution.probs == theta
        theta .= 0 # the kernel owns its probability snapshot
        @test distribution.probs[[1, end]] == [0.2, 0.8]
        samples = [rand(MersenneTwister(seed), distribution) for seed in 1:4000]
        @test all(sp -> sp == distribution.vals[1] || sp == distribution.vals[end], samples)
        @test count(==(distribution.vals[end]), samples) / length(samples) ≈ 0.8 atol=0.03
        wrong = DS.adaptive_state(model, DS.DSState((2, 3), 0))
        @test DS.pdf(distribution, wrong) == 0
        @test DS.pdf(distribution, DS.adaptive_state(model, r, nominal)) == 0
        # Every sibling has independent counts, including zero-mass branches.
        first_key = DS.CountKey(model, r, nominal, 4, r, nominal)
        distribution.vals[1].counts.increments[first_key] = 999
        @test isempty(s.counts.increments)
        @test !haskey(distribution.vals[end].counts.increments, first_key)
        # Explicit rows need normalization, nonnegative finite entries, and the
        # full 14-coordinate support; malformed inputs cannot create a kernel.
        for bad in (Float64[], ones(2), zeros(14), ones(14), fill(NaN, 14), fill(Inf, 14),
                    vcat([-0.1, 1.1], zeros(12)))
            @test_throws ArgumentError DS.adaptive_transition(model, s, 4, bad)
        end
    end

    @testset "Goal/unsafe entry counts and absorbing sources" begin
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)])
        cases = ((blocked, r, 4, DS.UNSAFE_CONFIG),
                 (model, DS.DSState((4, 5), 0), 4, DS.DSState(model.region_B)))
        for (p, config, action, target) in cases
            state = DS.adaptive_state(p, config)
            theta = zeros(14)
            theta[8] = 1 # commanded destination, unchanged nominal mode
            distribution = DS.adaptive_transition(p, state, action, theta)
            terminal = rand(MersenneTwister(4), distribution)
            @test terminal.config == target
            @test DS.is_terminal(terminal)
            key = DS.CountKey(p, config, nominal, action, target, nominal)
            @test terminal.counts.increments == Dict(key => 1)
            @test isempty(state.counts.increments)
            for a in 1:6
                absorbed = DS.adaptive_transition(p, terminal, a)
                @test rand(MersenneTwister(5), absorbed) === terminal
                @test DS.pdf(absorbed, terminal) == 1
                @test rand(DS.adaptive_transition(p, terminal, a, [1])) === terminal
            end
            @test_throws ArgumentError DS.posterior_predictive_row(p, terminal, action)
            @test_throws ArgumentError DS.adaptive_transition(p, terminal, action, [0.5, 0.5])
        end
    end

    @testset "HDR-constrained candidate kernels" begin
        # Uniform prior gives the full simplex, so vertices are admissible.
        hdr = DS.build_row_hdr(model, s, 4; epsilon=0.1, n_samples=20, rng=MersenneTwister(1))
        theta = zeros(14)
        theta[8] = 1
        robust = DS.robust_transition(model, s, 4, theta; hdr)
        chosen = rand(MersenneTwister(2), robust)
        @test chosen.config == DS.DSState((4, 3), 0)
        @test chosen.mode == nominal
        @test DS.pdf(robust, chosen) == 1
        @test length(chosen.counts.increments) == 1
        # HDR ordering is respected, not silently interpreted in default order.
        reordered = DS.build_row_hdr(model, s, 4, reverse(hdr.successors);
                                    epsilon=0.1, n_samples=20, rng=MersenneTwister(1))
        @test rand(MersenneTwister(2), DS.robust_transition(model, s, 4, reverse(theta); hdr=reordered)) == chosen
        @test_throws ArgumentError DS.robust_transition(model, s, 1, theta; hdr)
        incomplete = DS.build_row_hdr(model, s, 4, hdr.successors[1:2]; epsilon=0.1, n_samples=20)
        @test_throws ArgumentError DS.robust_transition(model, s, 4, [0.5, 0.5]; hdr=incomplete)
        key = DS.CountKey(model, r, nominal, 4, r, nominal)
        learned = DS.adaptive_state(model, r, nominal, DS.increment_count(model, s.counts, key))
        @test_throws ArgumentError DS.robust_transition(model, learned, 4, theta; hdr)

        updated_hdr = DS.build_row_hdr(model, learned, 4; epsilon=0.1, n_samples=1000,
                                      rng=MersenneTwister(3))
        outside = zeros(14)
        outside[8] = 1 # density zero under Dir(2,1,...,1)
        @test !(outside in updated_hdr)
        @test_throws ArgumentError DS.robust_transition(model, learned, 4, outside; hdr=updated_hdr)
        inside = zeros(14)
        inside[1] = 1 # maximum density, and this kernel increments the stay count
        stay = rand(DS.robust_transition(model, learned, 4, inside; hdr=updated_hdr))
        @test stay.config == r
        @test DS.count_value(model, stay.counts, key) == 3
        @test DS.count_value(model, learned.counts, key) == 2
        @test_throws ArgumentError DS.robust_transition(model, learned, 4, ones(14); hdr=updated_hdr)
        terminal = DS.adaptive_state(model, DS.UNSAFE_CONFIG)
        @test_throws ArgumentError DS.robust_transition(model, terminal, 4, theta; hdr)
    end

    @testset "Invalid source labels and actions" begin
        invalid_sources = (DS.AdaptiveState(r, nominal; outcome=DS.GOAL_STATE),
                           DS.AdaptiveState(DS.DSState((100, 100))))
        for source in invalid_sources
            @test_throws ArgumentError DS.posterior_predictive_row(model, source, 4)
            @test_throws ArgumentError DS.adaptive_transition(model, source, 4)
            @test_throws ArgumentError DS.adaptive_transition(model, source, 4, fill(1 / 14, 14))
        end
        for action in (0, 7)
            @test_throws ArgumentError DS.posterior_predictive_row(model, s, action)
            @test_throws ArgumentError DS.adaptive_transition(model, s, action)
            @test_throws ArgumentError DS.adaptive_transition(model, s, action, fill(1 / 14, 14))
        end
    end
end
