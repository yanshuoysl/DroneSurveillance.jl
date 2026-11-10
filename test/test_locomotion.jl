using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "AR locomotion and mode dynamics" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    one_fault = (true, false, false, false, false, false)
    all_faults = (true, true, true, true, true, true)
    r = DS.DSState((3, 3), 0)
    s = DS.adaptive_state(model, r)

    @testset "Uncertainty set then physical kernel, with no count update" begin
        for heading in (0, 90, 180, 270), (mode, n) in
                ((nominal, 14), (one_fault, 12), (all_faults, 2)), action in 1:6
            config = DS.DSState((3, 3), heading)
            state = DS.adaptive_state(model, config, mode)
            before = copy(state.counts.increments)
            hdr = DS.locomotion_mode_uncertainty(model, state, action; epsilon=0.1,
                                                n_samples=12, rng=MersenneTwister(40))
            @test hdr isa DS.DirichletHDR
            @test length(hdr.successors) == n
            @test hdr.successors == DS.feasible_successors(model, config, mode, action)
            @test hdr.distribution.alpha == fill(2, n)
            # The uncertain row is a set: two distinct joint rows are allowed.
            # Neither equals the posterior-mean row, even with all flags set.
            theta = fill(1 / n, n)
            theta[1] -= 0.01 / n
            theta[end] += 0.01 / n
            saved_theta = copy(theta)
            @test theta in hdr
            @test reverse(theta) in hdr
            saved_successors = copy(hdr.successors)
            saved_alpha = copy(hdr.distribution.alpha)
            saved_tau = hdr.tau
            physical = DS.locomotion_mode_transition(model, state, action, theta; hdr)
            @test physical.vals == saved_successors
            @test physical.probs ≈ theta
            @test sum(physical.probs) ≈ 1
            @test DS.pdf(physical, saved_successors[1]) ≈ theta[1]
            @test DS.pdf(physical, saved_successors[end]) ≈ theta[end]
            rng = MersenneTwister(41)
            for _ in 1:20
                pair = rand(rng, physical)
                @test pair isa Tuple{DS.DSState, DS.FaultMode}
                @test pair in saved_successors
            end
            @test state.counts.increments == before
            @test hdr.successors == saved_successors
            @test hdr.distribution.alpha == saved_alpha
            @test hdr.tau == saved_tau
            # Caller edits cannot change the already constructed kernel.
            theta .= 0
            hdr.successors[1] = saved_successors[end]
            @test physical.probs[1] ≈ saved_theta[1]
            @test physical.vals == saved_successors
        end
    end

    @testset "Current count row determines the set, not a fixed mean" begin
        key = DS.CountKey(model, r, nominal, 4, DS.DSState((4, 3), 0), nominal)
        other = DS.CountKey(model, r, nominal, 1, DS.DSState((3, 4), 0), nominal)
        phi = DS.DirichletCounts(Dict(key => 4, other => 9))
        state = DS.adaptive_state(model, r, nominal, phi)
        before = copy(phi.increments)
        hdr = DS.locomotion_mode_uncertainty(model, state, 4; epsilon=0.1,
                                            n_samples=1000, rng=MersenneTwister(42))
        @test hdr.distribution.alpha == vcat(fill(2, 7), [6], fill(2, 6))
        exponents = hdr.distribution.alpha .- 1
        theta = exponents ./ sum(exponents) # Dirichlet mode, distinct from its mean
        physical = DS.locomotion_mode_transition(model, state, 4, theta; hdr)
        @test DS.pdf(physical, (DS.DSState((4, 3), 0), nominal)) ≈ 5 / 18
        @test DS.count_value(model, state.counts, key) == 6
        @test phi.increments == before
        @test DS.pdf(physical, (DS.DSState((2, 3), 0), nominal)) == 0
        @test DS.pdf(physical, state) == 0 # outcomes contain no phi' field
        outside = zeros(14)
        outside[1] = 1 # every boundary coordinate has zero density under this prior
        @test_throws ArgumentError DS.locomotion_mode_transition(model, state, 4, outside; hdr)
        @test phi.increments == before
        @test_throws MethodError DS.locomotion_mode_transition(model, state, 4)
        @test_throws ArgumentError DS.locomotion_mode_transition(model, state, 4, theta)
    end

    @testset "Coordinate ordering and compatibility with existing composition" begin
        hdr = DS.locomotion_mode_uncertainty(model, s, 4; epsilon=0.1,
                                            n_samples=12, rng=MersenneTwister(43))
        theta = fill(1 / 14, 14)
        theta[1] -= 0.01 / 14
        theta[end] += 0.01 / 14
        physical = DS.locomotion_mode_transition(model, s, 4, theta; hdr)
        reversed_hdr = DS.build_row_hdr(model, s, 4, reverse(hdr.successors);
                                       epsilon=0.1, n_samples=12, rng=MersenneTwister(43))
        reordered = DS.locomotion_mode_transition(model, s, 4, reverse(theta); hdr=reversed_hdr)
        @test all(DS.pdf(reordered, pair) == DS.pdf(physical, pair) for pair in hdr.successors)
        # The combined kernel applies the separate parameter-update factor
        # after the physical factor. Its branch probabilities are unchanged.
        combined = DS.robust_transition(model, s, 4, theta; hdr)
        @test all(DS.pdf(physical, (sp.config, sp.mode)) == probability
                  for (sp, probability) in zip(combined.vals, combined.probs))
        @test isempty(s.counts.increments)
    end

    @testset "Reject malformed, stale, and unrelated rows" begin
        hdr = DS.locomotion_mode_uncertainty(model, s, 4; epsilon=0.1,
                                            n_samples=12, rng=MersenneTwister(44))
        theta = fill(1 / 14, 14)
        @test_throws ArgumentError DS.locomotion_mode_transition(model, s, 1, theta; hdr)
        incomplete = DS.build_row_hdr(model, s, 4, hdr.successors[1:2]; epsilon=0.1, n_samples=12)
        @test_throws ArgumentError DS.locomotion_mode_transition(model, s, 4, [0.5, 0.5]; hdr=incomplete)
        key = DS.CountKey(model, r, nominal, 4, r, nominal)
        different_counts = DS.adaptive_state(model, r, nominal, DS.DirichletCounts(Dict(key => 1)))
        @test_throws ArgumentError DS.locomotion_mode_transition(model, different_counts, 4, theta; hdr)
        bad_support = copy(hdr.successors)
        bad_support[2] = (DS.DSState((2, 3), 0), nominal)
        unrelated = DS.DirichletHDR(hdr.distribution, bad_support, hdr.epsilon, hdr.n_samples, hdr.tau)
        @test_throws ArgumentError DS.locomotion_mode_transition(model, s, 4, theta; hdr=unrelated)
        for threshold in (NaN, Inf, -1.0, 0.0)
            invalid = DS.DirichletHDR(hdr.distribution, copy(hdr.successors), hdr.epsilon,
                                      hdr.n_samples, threshold)
            @test_throws ArgumentError DS.locomotion_mode_transition(model, s, 4, theta; hdr=invalid)
        end
        for bad in (Float64[], zeros(14), ones(14), fill(NaN, 14), fill(Inf, 14),
                    vcat([-0.1, 1.1], zeros(12)))
            @test_throws ArgumentError DS.locomotion_mode_transition(model, s, 4, bad; hdr)
        end
        invalid_source = DS.AdaptiveState(r, nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.locomotion_mode_uncertainty(model, invalid_source, 4; epsilon=0.1)
        @test_throws ArgumentError DS.locomotion_mode_transition(model, invalid_source, 4, theta; hdr)
        for action in (0, 7)
            @test_throws ArgumentError DS.locomotion_mode_uncertainty(model, s, action; epsilon=0.1)
            @test_throws ArgumentError DS.locomotion_mode_transition(model, s, action, theta; hdr)
        end
    end

    @testset "Unsafe and goal configurations, without parameter updates" begin
        for (p, source, target) in (
                (DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)]), r, DS.UNSAFE_CONFIG),
                (model, DS.DSState((4, 5), 0), DS.DSState(model.region_B)))
            state = DS.adaptive_state(p, source)
            hdr = DS.locomotion_mode_uncertainty(p, state, 4; epsilon=0.1,
                                                n_samples=12, rng=MersenneTwister(45))
            theta = fill(1 / 14, 14)
            physical = DS.locomotion_mode_transition(p, state, 4, theta; hdr)
            @test DS.pdf(physical, (target, nominal)) ≈ 1 / 14
            vertex = zeros(14)
            vertex[8] = 1
            @test_throws ArgumentError DS.locomotion_mode_transition(p, state, 4, vertex; hdr)
            @test isempty(state.counts.increments)
        end
        for config in (DS.UNSAFE_CONFIG, DS.DSState(model.region_B)),
                mode in (nominal, one_fault, all_faults), action in 1:6
            state = DS.adaptive_state(model, config, mode)
            physical = DS.locomotion_mode_transition(model, state, action, [1])
            @test rand(physical) == (config, mode)
            @test DS.pdf(physical, (config, mode)) == 1
            @test isempty(state.counts.increments)
            @test_throws ArgumentError DS.locomotion_mode_uncertainty(model, state, action; epsilon=0.1)
            @test_throws ArgumentError DS.locomotion_mode_transition(model, state, action, [0.5, 0.5])
        end
    end
end
