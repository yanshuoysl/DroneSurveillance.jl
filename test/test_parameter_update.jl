using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "AR parameter-update transition" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    one_fault = (true, false, false, false, false, false)
    all_faults = (true, true, true, true, true, true)
    r = DS.DSState((3, 3), 0)
    other_key = DS.CountKey(model, DS.DSState((2, 2), 90), nominal, 5,
                           DS.DSState((2, 2), 180), nominal)

    @testset "Unit delta and exact 1/0 mass for every conditioned branch" begin
        for heading in (0, 90, 180, 270), mode in (nominal, one_fault, all_faults), action in 1:6
            config = DS.DSState((3, 3), heading)
            successors = DS.feasible_successors(model, config, mode, action)
            keys = [DS.CountKey(model, config, mode, action, rp, mp) for (rp, mp) in successors]
            increments = Dict(key => j - 1 for (j, key) in enumerate(keys))
            increments[other_key] = 7 # an unrelated source/action row
            phi = DS.DirichletCounts(increments)
            before = copy(phi.increments)
            state = DS.adaptive_state(model, config, mode, phi)
            for (j, (rp, mp)) in enumerate(successors)
                conditional = DS.parameter_update_transition(model, state, action, rp, mp)
                updated = rand(MersenneTwister(7), conditional)
                expected = copy(before)
                expected[keys[j]] = j # old stored increment j-1, plus one
                @test updated isa DS.DirichletCounts
                @test updated.increments == expected
                @test updated.increments !== phi.increments
                @test DS.pdf(conditional, DS.DirichletCounts(expected)) == 1.0
                @test DS.pdf(conditional, phi) == 0.0
                @test DS.count_value(model, updated, keys[j]) == j + 1
                @test DS.count_value(model, updated, other_key) == 8
                # A different delta or a double increment has probability zero.
                wrong = copy(before)
                alternative = mod1(j + 1, length(keys))
                wrong[keys[alternative]] = get(wrong, keys[alternative], 0) + 1
                @test DS.pdf(conditional, DS.DirichletCounts(wrong)) == 0.0
                doubled = copy(expected)
                doubled[keys[j]] += 1
                @test DS.pdf(conditional, DS.DirichletCounts(doubled)) == 0.0
                @test phi.increments == before
                # Reading or sampling this deterministic factor again does not
                # perform another increment. The explicit-factor API agrees.
                @test rand(conditional) === updated
                explicit = DS.parameter_update_transition(model, config, mode, phi, action, rp, mp)
                @test rand(explicit) == updated
            end
        end
    end

    @testset "Separate histories and action-specific counts" begin
        state = DS.adaptive_state(model, r)
        right = DS.DSState((4, 3), 0)
        move_key = DS.CountKey(model, r, nominal, 4, right, nominal)
        stay_key = DS.CountKey(model, r, nominal, 4, r, nominal)
        moved = rand(DS.parameter_update_transition(model, state, 4, right, nominal))
        stayed = rand(DS.parameter_update_transition(model, state, 4, r, nominal))
        @test moved.increments == Dict(move_key => 1)
        @test stayed.increments == Dict(stay_key => 1)
        @test DS.count_row(model, moved, r, nominal, 1) == ones(Int, 14)
        moved.increments[move_key] = 99
        @test stayed.increments == Dict(stay_key => 1)
        @test isempty(state.counts.increments)
    end

    @testset "Factorization agrees with the full robust kernel" begin
        for mode in (nominal, one_fault), action in 1:6
            state = DS.adaptive_state(model, r, mode)
            hdr = DS.locomotion_mode_uncertainty(model, state, action; epsilon=0.1,
                                                n_samples=12, rng=MersenneTwister(action))
            theta = zeros(length(hdr.successors))
            theta[1], theta[end] = 0.25, 0.75
            physical = DS.locomotion_mode_transition(model, state, action, theta; hdr)
            combined = DS.robust_transition(model, state, action, theta; hdr)
            @test sum(combined.probs) == 1.0
            for sp in combined.vals
                parameters = DS.parameter_update_transition(model, state, action, sp.config, sp.mode)
                @test DS.pdf(combined, sp) ==
                      DS.pdf(physical, (sp.config, sp.mode)) * DS.pdf(parameters, sp.counts)
                wrong_history = DS.adaptive_state(model, sp.config, sp.mode, state.counts)
                @test DS.pdf(combined, wrong_history) == 0.0
                @test DS.pdf(parameters, wrong_history.counts) == 0.0
            end
            @test isempty(state.counts.increments)
            # A stay/unchanged-mode branch updates this same source row; the
            # old uncertainty set must not be silently reused after that update.
            updated_phi = rand(DS.parameter_update_transition(model, state, action, r, mode))
            updated_state = DS.adaptive_state(model, r, mode, updated_phi)
            @test_throws ArgumentError DS.locomotion_mode_transition(model, updated_state,
                                                                    action, theta; hdr)
        end
    end

    @testset "Terminal entry increments once; absorbing source does not" begin
        for (p, config, target) in (
                (DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)]), r, DS.UNSAFE_CONFIG),
                (model, DS.DSState((4, 5), 0), DS.DSState(model.region_B)))
            source = DS.adaptive_state(p, config)
            conditional = DS.parameter_update_transition(p, source, 4, target, nominal)
            entered = rand(conditional)
            key = DS.CountKey(p, config, nominal, 4, target, nominal)
            @test entered.increments == Dict(key => 1)
            terminal = DS.adaptive_state(p, target, nominal, entered)
            @test DS.is_terminal(terminal)
            for action in 1:6
                absorbing = DS.parameter_update_transition(p, terminal, action, target, nominal)
                unchanged = rand(absorbing)
                @test unchanged == entered
                @test unchanged.increments !== entered.increments
                @test DS.pdf(absorbing, entered) == 1.0
                @test_throws ArgumentError DS.parameter_update_transition(p, terminal, action,
                                                                          target, one_fault)
                @test_throws ArgumentError DS.parameter_update_transition(p, terminal, action,
                                                                          DS.DSState((2, 2)), nominal)
            end
            @test isempty(source.counts.increments)
        end
    end

    @testset "Invalid conditioning and count overflow are rejected" begin
        state = DS.adaptive_state(model, r)
        right = DS.DSState((4, 3), 0)
        for wrong in (DS.DSState((2, 3), 0), DS.DSState((3, 4), 0),
                      DS.DSState((4, 3), 90), DS.DSState((5, 3), 0), DS.UNSAFE_CONFIG)
            @test_throws ArgumentError DS.parameter_update_transition(model, state, 4, wrong, nominal)
        end
        @test_throws ArgumentError DS.parameter_update_transition(model, state, 4, right,
                                                                 (true, true, false, false, false, false))
        failed_state = DS.adaptive_state(model, r, one_fault)
        @test_throws ArgumentError DS.parameter_update_transition(model, failed_state, 4, right, nominal)
        for action in (0, 7)
            @test_throws ArgumentError DS.parameter_update_transition(model, state, action, right, nominal)
        end
        mislabeled = DS.AdaptiveState(r, nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.parameter_update_transition(model, mislabeled, 4, right, nominal)
        @test_throws ArgumentError DS.parameter_update_transition(model, DS.DSState((100, 100)), nominal,
                                                                 state.counts, 4, right, nominal)
        key = DS.CountKey(model, r, nominal, 4, right, nominal)
        maximum = DS.DirichletCounts(Dict(key => typemax(Int) - 1))
        large_state = DS.adaptive_state(model, r, nominal, maximum)
        @test_throws OverflowError DS.parameter_update_transition(model, large_state, 4, right, nominal)
        @test maximum.increments[key] == typemax(Int) - 1
        @test isempty(state.counts.increments)
    end
end
