using Test
using Random
using POMDPs
using POMDPTools
using Distributions: logpdf

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Exact conditional AR belief update" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2)])
    free = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    o = DS.DSObservation(3, 1, 1)
    s0 = DS.initial_adaptive_state(p)

    function chosen_rows(p, b, a)
        rows = Dict{DS.AdaptiveState, DS.SelectedTransitionRow}()
        rng = MersenneTwister(42)
        for s in support(b)
            DS.is_terminal(s) && continue
            hdr = DS.locomotion_mode_uncertainty(p, s, a;
                epsilon=0.1, n_samples=512, rng=rng)
            theta = fill(1 / length(hdr.successors), length(hdr.successors))
            rows[s] = DS.SelectedTransitionRow(theta, hdr)
        end
        return DS.BeliefTransitionRows(p, b, a, rows)
    end

    @testset "Finite initialization, merging, validation, and immutable snapshots" begin
        b = DS.ExactAdaptiveBelief(p)
        @test length(b) == 1
        @test pdf(b, s0) == 1
        @test only(support(b)) == s0
        @test DS.count_row(p, only(support(b)).counts, s0.config, nominal, 1) == fill(2, 14)
        @test length(DS.ExactAdaptiveBelief(p, [s0 => 0.2, s0 => 0.8])) == 1
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, [s0 => -1.0])
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, [s0 => Inf])
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, [s0 => NaN])
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, [s0 => 0.0])
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, Pair{DS.AdaptiveState,Float64}[])
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, [s0 => Inf]; logweights=true)
        malformed = DS.AdaptiveState(s0.config, nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.ExactAdaptiveBelief(p, malformed)
        key = DS.CountKey(p, s0.config, nominal, 1, s0.config, nominal)
        source = DS.adaptive_state(p, s0.config, nominal, DS.DirichletCounts(Dict(key => 4)))
        expected = deepcopy(source)
        frozen = DS.ExactAdaptiveBelief(p, source)
        source.counts.increments[key] = 99
        @test pdf(frozen, expected) == 1
        @test pdf(frozen, source) == 0
        exposed = only(support(frozen))
        exposed.counts.increments[key] = 55
        @test pdf(frozen, expected) == 1
        iterated, w = only(collect(frozen))
        iterated.counts.increments[key] = 66
        @test w == 1 && pdf(frozen, expected) == 1
        @test only(support(frozen)).counts.increments[key] == 4
    end

    @testset "Worked one- and two-step posteriors" begin
        b0 = DS.ExactAdaptiveBelief(p)
        rows0 = chosen_rows(p, b0, 1)
        first_update = DS.exact_belief_update(p, b0, 1, o, rows0; return_info=true)
        b1 = first_update.belief
        @test length(b1) == 14
        @test first_update.parents == 1 && first_update.expanded_branches == 14
        @test exp(first_update.log_evidence) ≈ 0.378
        @test sum(w for (_, w) in b1) ≈ 1
        @test all(pdf(b1, s) ≈ (s.config.quad[2] == 1 ? 1 / 196 : 27 / 196) for s in support(b1))
        @test all(sum(values(s.counts.increments)) == 1 for s in support(b1))
        @test all(DS.count_value(s.counts, only(keys(s.counts.increments))) == 3 for s in support(b1))
        @test isempty(only(support(b0)).counts.increments)
        rows1 = chosen_rows(p, b1, 1)
        second_update = DS.exact_belief_update(p, b1, 1, o, rows1; return_info=true)
        b2 = second_update.belief
        @test length(b2) == 172
        @test second_update.parents == 14 && second_update.expanded_branches == 172
        @test exp(second_update.log_evidence) ≈ 0.7164642857142857
        for (y, size, probability) in ((1, 43, 1 / 1486), (2, 86, 756 / 1486), (3, 43, 729 / 1486))
            states = filter(s -> s.config.quad[2] == y, support(b2))
            @test length(states) == size
            @test sum(pdf(b2, s) for s in states) ≈ probability
        end
        @test all(sum(values(s.counts.increments)) == 2 for s in support(b2))
        @test all(sum(values(s.counts.increments)) == 1 for s in support(b1))
        @test !any(pdf(b2, s) > 0 for s in support(b1))
        # Four nominal physical paths retain their different full count histories.
        r0, r1, r2 = s0.config, DS.DSState((1, 2)), DS.DSState((1, 3))
        for (middle, final, expected) in ((r0, r0, 1 / 72814),
                (r0, r1, 27 / 72814), (r1, r1, 729 / 72814), (r1, r2, 729 / 72814))
            phi1 = rand(DS.parameter_update_transition(p, s0, 1, middle, nominal))
            parent = DS.adaptive_state(p, middle, nominal, phi1)
            phi2 = rand(DS.parameter_update_transition(p, parent, 1, final, nominal))
            child = DS.adaptive_state(p, final, nominal, phi2)
            @test pdf(b2, child) ≈ expected
        end
    end

    @testset "Full-state merging and separate count histories" begin
        rA, rB = DS.DSState((3, 3)), DS.DSState((4, 3))
        edgeA = DS.CountKey(free, rA, nominal, 4, rB, nominal)
        edgeB = DS.CountKey(free, rB, nominal, 4, rB, nominal)
        sA = DS.adaptive_state(free, rA, nominal, DS.DirichletCounts(Dict(edgeB => 1)))
        sB = DS.adaptive_state(free, rB, nominal, DS.DirichletCounts(Dict(edgeA => 1)))
        b = DS.ExactAdaptiveBelief(free, [sA => 0.4, sB => 0.6])
        rows = chosen_rows(free, b, 4)
        obs = DS.DSObservation(1, 1, 1)
        posterior = DS.exact_belief_update(free, b, 4, obs, rows)
        # Two positive-mass branches meet at one complete child. All other
        # children remain present because an all-two HDR excludes point rows.
        @test length(posterior) == 27
        merged = DS.adaptive_state(free, rB, nominal, DS.DirichletCounts(Dict(edgeA => 1, edgeB => 1)))
        likelihoodA = pdf(observation(free, 4, DS.adaptive_state(free, rA)), obs)
        likelihoodB = pdf(observation(free, 4, DS.adaptive_state(free, rB)), obs)
        likelihoodC = pdf(observation(free, 4, DS.adaptive_state(free, DS.DSState((5, 3)))), obs)
        evidence = 0.4 * (likelihoodA + likelihoodB) / 2 + 0.6 * (likelihoodB + likelihoodC) / 2
        @test pdf(posterior, merged) ≈ likelihoodB / (14 * evidence)
        @test sum(w for (_, w) in posterior) ≈ 1
        sPlain = DS.adaptive_state(free, rB)
        different = DS.ExactAdaptiveBelief(free, [sPlain => 0.3, sB => 0.7])
        rows = chosen_rows(free, different, 4)
        posterior = DS.exact_belief_update(free, different, 4, DS.DSObservation(1, 1, 1), rows)
        @test length(posterior) == 28
        @test sum(w for (s, w) in posterior if !haskey(s.counts.increments, edgeA)) ≈ 0.3
        @test sum(w for (s, w) in posterior if get(s.counts.increments, edgeA, 0) == 1) ≈ 0.7
    end

    @testset "Selected model snapshots, row validation, and updater lifecycle" begin
        up = DS.ExactAdaptiveUpdater(p)
        b = initialize_belief(up, nothing)
        @test length(b) == 1 && pdf(b, s0) == 1
        rows = chosen_rows(p, b, 1)
        @test_throws ArgumentError update(up, b, 1, o)
        @test_throws ArgumentError DS.BeliefTransitionRows(p, b, 1, Dict())
        @test_throws ArgumentError DS.exact_belief_update(p, b, 4, o, rows)
        @test_throws ArgumentError DS.exact_belief_update(p, b, 1, DS.DSObservation(5,1,1), rows)
        DS.prepare_belief_update!(up, b, 1, rows)
        b1 = update(up, b, 1, o)
        @test length(b1) == 14
        @test_throws ArgumentError update(up, b, 1, o)
        @test_throws ArgumentError DS.exact_belief_update(p, b1, 1, o, rows)
        @test pdf(initialize_belief(up, initialstate(p)), s0) == 1
        @test_throws ArgumentError initialize_belief(up, Deterministic(s0.config))
        @test pdf(initialize_belief(up, Deterministic(s0)), s0) == 1
        @test length(initialize_belief(up, b1)) == 14

        hdr = DS.locomotion_mode_uncertainty(p, s0, 1; epsilon=0.1, n_samples=512, rng=MersenneTwister(8))
        theta = fill(1 / 14, 14)
        selected = DS.SelectedTransitionRow(theta, hdr)
        dictionary = Dict(s0 => selected)
        prepared = DS.BeliefTransitionRows(p, b, 1, dictionary)
        theta[1] = 0
        hdr.successors[1] = hdr.successors[end]
        hdr.distribution.alpha[1] = 100
        empty!(dictionary)
        @test length(DS.exact_belief_update(p, b, 1, o, prepared)) == 14
        @test_throws ArgumentError DS.SelectedTransitionRow(zeros(14), hdr)
        # Reusing the same physical row HDR after changing that row's counts fails.
        stay_key = DS.CountKey(p, s0.config, nominal, 1, s0.config, nominal)
        learned = DS.adaptive_state(p, s0.config, nominal, DS.DirichletCounts(Dict(stay_key => 1)))
        changed = DS.ExactAdaptiveBelief(p, learned)
        @test_throws ArgumentError DS.BeliefTransitionRows(p, changed, 1, Dict(learned => selected))
        # A valid HDR can be reordered; theta must follow that same ordering.
        fresh = DS.locomotion_mode_uncertainty(p, s0, 1; epsilon=0.1, n_samples=512, rng=MersenneTwister(21))
        reversed = DS.DirichletHDR(fresh.distribution, reverse(fresh.successors), fresh.epsilon, fresh.n_samples, fresh.tau)
        interior = fill(1 / 14, 14)
        interior[1] -= 0.01 / 14
        interior[end] += 0.01 / 14
        rows = Dict(s0 => DS.SelectedTransitionRow(reverse(interior), reversed))
        posterior = DS.exact_belief_update(p, b, 1, o, rows)
        reference = DS.exact_belief_update(p, b, 1, o, Dict(s0 => DS.SelectedTransitionRow(interior, fresh)))
        @test length(posterior) == 14
        @test all(pdf(posterior, s) ≈ pdf(reference, s) for s in support(reference))
        vertex = zeros(14)
        vertex[1] = 1
        @test_throws ArgumentError DS.SelectedTransitionRow(vertex, fresh)
    end

    @testset "Tiny boundary hypotheses retain probability without vetoing actions" begin
        rare = DS.adaptive_state(free, DS.DSState((1, 3)))
        common = DS.adaptive_state(free, DS.DSState((3, 3)))
        b = DS.ExactAdaptiveBelief(free, [rare => -1000.0, common => 0.0]; logweights=true)
        @test length(b) == 2 && pdf(b, rare) == 0 && isfinite(logpdf(b, rare))
        @test actions(free,b) == collect(1:6)
        @test_throws ArgumentError DS.BeliefTransitionRows(free, b, 3, Dict())
        rows = chosen_rows(free, b, 3)
        posterior = DS.exact_belief_update(free, b, 3, DS.DSObservation(1, 1, 1), rows)
        @test length(posterior) == 21
        @test all(isfinite(logpdf(posterior, s)) for s in support(posterior))
        @test minimum(logpdf(posterior, s) for s in support(posterior)) < -700
        @test actions(free,posterior) == collect(1:6)
        collision = DS.exact_belief_update(free,b,3,DS.TERMINAL_OBSERVATION,rows)
        @test length(collision) == 7 && all(DS.is_unsafe(s) for s in support(collision))
        @test all(sum(values(s.counts.increments)) == 1 for s in support(collision))
    end

    @testset "Zero evidence and underflow-safe camera likelihood" begin
        perfect = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[], camera=DS.KnownCamera(correct_probability=1))
        source = DS.adaptive_state(perfect, DS.DSState((3, 3)))
        b = DS.ExactAdaptiveBelief(perfect, source)
        rows = chosen_rows(perfect, b, 4)
        up = DS.ExactAdaptiveUpdater(perfect)
        DS.prepare_belief_update!(up, b, 4, rows)
        @test_throws DS.ZeroObservationEvidence update(up, b, 4, DS.DSObservation(4,4,4))
        @test length(update(up, b, 4, DS.DSObservation(1,1,1))) == 14
        confusion = fill(1e-200, 4, 4)
        for i in 1:4
            confusion[i, i] = 1
        end
        tiny = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[], camera=DS.KnownCamera(confusion))
        b = DS.ExactAdaptiveBelief(tiny, source)
        rows = chosen_rows(tiny, b, 4)
        @test pdf(observation(tiny, 4, source), DS.DSObservation(4,4,4)) == 0
        result = DS.exact_belief_update(tiny, b, 4, DS.DSObservation(4,4,4), rows; return_info=true)
        @test length(result.belief) == 14
        @test result.log_evidence ≈ 3log(1e-200)
    end

    @testset "Goal/collision entry and terminal absorption" begin
        for obstacles in (DS.DSPos[], [DS.DSPos(2, 1)])
            model = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(2, 1) in obstacles ? DS.DSPos(5,5) : DS.DSPos(2,1), obstacles=obstacles)
            b = DS.ExactAdaptiveBelief(model)
            rows = chosen_rows(model, b, 4)
            entered = DS.exact_belief_update(model, b, 4, DS.TERMINAL_OBSERVATION, rows)
            @test length(entered) == 7
            @test all(isterminal(model, s) for s in support(entered))
            @test all(sum(values(s.counts.increments)) == 1 for s in support(entered))
            @test isempty(actions(model, entered))
            rows = DS.BeliefTransitionRows(model, entered, 5, Dict())
            absorbed = DS.exact_belief_update(model, entered, 5, DS.TERMINAL_OBSERVATION, rows)
            @test length(absorbed) == 7
            @test all(pdf(absorbed, s) ≈ pdf(entered, s) for s in support(entered))
            @test all(sum(values(s.counts.increments)) == 1 for s in support(absorbed))
        end
    end
end
