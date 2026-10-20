using Test
using POMDPs
using POMDPTools
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Adaptive boundary action availability" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    modes = (nominal, (true, false, false, false, false, false),
             (true, true, true, true, true, true))
    # Independent command displacements, ordered by heading then action ID.
    offsets = (
        ((0, 1), (0, -1), (-1, 0), (1, 0)),
        ((1, 0), (-1, 0), (0, 1), (0, -1)),
        ((0, -1), (0, 1), (1, 0), (-1, 0)),
        ((-1, 0), (1, 0), (0, -1), (0, 1)),
    )
    @test collect(actions(model)) == collect(1:6)
    for x in 1:5, y in 1:5, (hi, heading) in enumerate((0, 90, 180, 270))
        r = DS.DSState((x, y), heading)
        expected = Int64[a for a in 1:6 if a > 4 ||
            (1 <= x + offsets[hi][a][1] <= 5 && 1 <= y + offsets[hi][a][2] <= 5)]
        @test actions(model, r) == expected
        for mode in modes
            s = DS.adaptive_state(model, r, mode)
            @test actions(model, s) == (DS.is_terminal(s) ? Int64[] : expected)
            DS.is_terminal(s) && continue
            for a in 1:6
                if a in expected
                    physical = DS.feasible_successors(model, r, mode, a)
                    @test length(physical) == 2 * (7 - count(identity, mode))
                    @test all(pair -> pair[1] != DS.UNSAFE_CONFIG, physical)
                    @test all(sp -> 1 <= sp.config.quad[1] <= 5 &&
                                   1 <= sp.config.quad[2] <= 5,
                              DS.adaptive_transition(model, s, a).vals)
                else
                    @test_throws ArgumentError transition(model, r, a)
                    @test_throws ArgumentError DS.feasible_successors(model, r, mode, a)
                    @test !DS.is_feasible_successor(model, r, mode, a, r, mode)
                    @test !DS.is_feasible_successor(model, r, mode, a, DS.UNSAFE_CONFIG, mode)
                    @test_throws ArgumentError DS.CountKey(model, r, mode, a, r, mode)
                    @test_throws ArgumentError DS.count_row(model, s.counts, r, mode, a)
                    @test_throws ArgumentError DS.adaptive_transition(model, s, a)
                    @test_throws ArgumentError DS.parameter_update_transition(model, s, a, r, mode)
                    @test isempty(s.counts.increments)
                end
            end
        end
    end

    @testset "An outward command cannot acquire or use an uncertainty row" begin
        interior = DS.adaptive_state(model, DS.DSState((3, 3)))
        hdr = DS.locomotion_mode_uncertainty(model, interior, 4;
                                            epsilon=0.1, n_samples=1)
        theta = fill(1 / 14, 14)
        for (a, position) in enumerate(((3, 5), (3, 1), (1, 3), (5, 3)))
            s = DS.adaptive_state(model, DS.DSState(position))
            @test_throws ArgumentError DS.locomotion_mode_uncertainty(model, s, a;
                                                                     epsilon=0.1, n_samples=1)
            @test_throws ArgumentError DS.locomotion_mode_transition(model, s, a, theta; hdr)
            @test_throws ArgumentError DS.robust_transition(model, s, a, theta; hdr)
            @test isempty(s.counts.increments)
        end
    end

    @testset "Corners, narrow grids, obstacles, and terminal configurations" begin
        @test actions(model, DS.DSState((1, 1))) == [1, 4, 5, 6]
        @test actions(model, DS.DSState((5, 3))) == [1, 2, 3, 5, 6]
        # Identical availability despite every possible fault combination.
        for mode in DS.all_fault_modes()
            @test actions(model, DS.adaptive_state(model, DS.DSState((1, 1)), mode)) == [1, 4, 5, 6]
        end
        narrow = DS.DroneSurveillancePOMDP(size=(1, 3), obstacles=DS.DSPos[])
        @test actions(narrow, DS.adaptive_state(narrow, DS.DSState((1, 2)))) == [1, 2, 5, 6]
        @test actions(narrow, DS.adaptive_state(narrow, DS.DSState((1, 2), 90))) == [3, 4, 5, 6]
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 1)])
        source = DS.adaptive_state(blocked, DS.DSState((1, 1)))
        @test 4 in actions(blocked, source)
        @test DS.is_feasible_successor(blocked, source.config, nominal, 4, DS.UNSAFE_CONFIG, nominal)
        @test isempty(actions(model, DS.adaptive_state(model, DS.DSState(model.region_B))))
        @test isempty(actions(model, DS.adaptive_state(model, DS.UNSAFE_CONFIG)))
        @test isempty(actions(model, model.terminal_state))
    end

    @testset "Belief action selection respects every possible configuration" begin
        probabilities = zeros(length(model))
        left = DS.DSState((1, 3))
        right = DS.DSState((5, 3))
        probabilities[stateindex(model, left)] = 0.5
        probabilities[stateindex(model, right)] = 0.5
        b = DiscreteBelief(model, probabilities)
        @test actions(model, b) == [1, 2, 5, 6]
        policy = RandomPolicy(model; rng=MersenneTwister(47), updater=DiscreteUpdater(model))
        @test all(action(policy, b) in [1, 2, 5, 6] for _ in 1:30)
        probabilities[stateindex(model, left)] = 0
        probabilities[stateindex(model, right)] = 1
        @test actions(model, DiscreteBelief(model, probabilities)) == [1, 2, 3, 5, 6]
        fill!(probabilities, 0)
        probabilities[stateindex(model, model.terminal_state)] = 1
        @test isempty(actions(model, DiscreteBelief(model, probabilities)))
    end
end
