using Test
using Random

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Adaptive successor support" begin
    DS = DroneSurveillanceAdaptive
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    nominal = DS.NOMINAL_MODE
    modes = DS.all_fault_modes()

    @testset "Every action, heading, and fault mode" begin
        # Independent absolute-motion oracle: forward, backward, left, right.
        translations = (
            ((3, 4), (3, 2), (2, 3), (4, 3)), # north
            ((4, 3), (2, 3), (3, 4), (3, 2)), # east
            ((3, 2), (3, 4), (4, 3), (2, 3)), # south
            ((2, 3), (4, 3), (3, 2), (3, 4)), # west
        )
        for (hi, heading) in enumerate((0, 90, 180, 270))
            r = DS.DSState((3, 3), heading)
            commanded = vcat([DS.DSState(p, heading) for p in translations[hi]],
                             [DS.DSState(r.quad, mod(heading + 90, 360)),
                              DS.DSState(r.quad, mod(heading - 90, 360))])
            for mode in modes, action in 1:6
                admissible_modes = [mp for mp in modes
                    if all(!mode[i] || mp[i] for i in 1:6) &&
                       count(identity, mp) - count(identity, mode) <= 1]
                expected = [(rp, mp) for rp in (r, commanded[action]) for mp in admissible_modes]
                actual = DS.feasible_successors(model, r, mode, action)
                @test actual == expected
                @test length(unique(actual)) == 2 * (7 - count(identity, mode))
                @test all(count(pair -> pair[2] == mp, actual) == 2 for mp in admissible_modes)
                @test all(DS.is_feasible_successor(model, r, mode, action, rp, mp)
                          for (rp, mp) in actual)
                # Both stay and commanded motion are retained for each mode;
                # other directions and unintended headings remain impossible.
                for candidate in vcat([r], commanded)
                    @test DS.is_feasible_successor(model, r, mode, action, candidate, mode) ==
                          (candidate == r || candidate == commanded[action])
                end
            end
        end
    end

    @testset "Right-action structural zeros and count isolation" begin
        r = DS.DSState((3, 3), 0)
        right = DS.DSState((4, 3), 0)
        failed_right = (false, false, false, true, false, false)
        @test DS.is_feasible_successor(model, r, nominal, 4, right, nominal)
        @test DS.is_feasible_successor(model, r, nominal, 4, r, failed_right)
        @test DS.is_feasible_successor(model, r, nominal, 4, right, failed_right)
        @test DS.is_feasible_successor(model, r, nominal, 4, r, nominal)
        @test DS.is_feasible_successor(model, r, failed_right, 4, right, failed_right)
        @test DS.is_feasible_successor(model, r, failed_right, 4, r, failed_right)
        @test !DS.is_feasible_successor(model, r, failed_right, 4, right, nominal)
        @test !DS.is_feasible_successor(model, r, nominal, 4, r,
                                      (true, false, false, true, false, false))

        phi = DS.DirichletCounts()
        key = DS.CountKey(model, r, nominal, 4, right, nominal)
        updated = DS.increment_count(model, phi, key)
        @test DS.count_value(model, phi, key) == 1
        @test DS.count_value(model, updated, key) == 2
        @test isempty(phi.increments)
        @test DS.count_row(model, updated, r, nominal, 4) == vcat(ones(Int, 7), [2], ones(Int, 6))
        @test DS.count_row(model, updated, r, nominal, 1) == ones(Int, 14)
        support = DS.feasible_successors(model, r, nominal, 4)
        @test DS.count_row(model, updated, r, nominal, 4, reverse(support)) ==
              vcat(ones(Int, 6), [2], ones(Int, 7))
        @test_throws ArgumentError DS.count_row(model, phi, r, nominal, 4, support[1:2])
        @test_throws ArgumentError DS.count_row(model, phi, r, nominal, 4,
                                               vcat(support[1:end-1], support[1:1]))
        # Updating one cross-combination must not update other modes or cells.
        stay_key = DS.CountKey(model, r, nominal, 4, r, nominal)
        flagged_key = DS.CountKey(model, r, nominal, 4, right, failed_right)
        @test DS.count_value(model, updated, stay_key) == 1
        @test DS.count_value(model, updated, flagged_key) == 1
        cross_updated = DS.increment_count(model, updated, flagged_key)
        @test DS.count_value(model, cross_updated, key) == 2
        @test DS.count_value(model, cross_updated, flagged_key) == 2
        @test DS.count_value(model, cross_updated, stay_key) == 1
        # Even an unchecked storage key cannot enter the map-aware count APIs.
        for wrong in (DS.DSState((2, 3), 0), DS.DSState((3, 4), 0),
                      DS.DSState((3, 2), 0), DS.DSState((4, 3), 90),
                      DS.DSState((5, 3), 0), DS.UNSAFE_CONFIG)
            @test !DS.is_feasible_successor(model, r, nominal, 4, wrong, nominal)
            @test_throws ArgumentError DS.CountKey(model, r, nominal, 4, wrong, nominal)
            raw_key = DS.CountKey(r, nominal, 4, wrong, nominal)
            @test_throws ArgumentError DS.increment_count(model, phi, raw_key)
            @test_throws ArgumentError DS.count_value(model, phi, raw_key)
            bad_support = copy(support)
            bad_support[1] = (wrong, nominal)
            @test_throws ArgumentError DS.count_row(model, phi, r, nominal, 4, bad_support)
        end
        @test isempty(phi.increments)
    end

    @testset "Unsafe entries and absorption" begin
        r = DS.DSState((3, 3), 0)
        for (action, obstacle) in enumerate(((3, 4), (3, 2), (2, 3), (4, 3)))
            blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(obstacle)])
            failed = ntuple(i -> i == action, 6)
            @test DS.is_feasible_successor(blocked, r, nominal, action, DS.UNSAFE_CONFIG, nominal)
            @test !DS.is_feasible_successor(blocked, r, nominal, action,
                                           DS.DSState(obstacle), nominal)
            @test DS.is_feasible_successor(blocked, r, nominal, action, r, failed)
            @test DS.is_feasible_successor(blocked, r, nominal, action, DS.UNSAFE_CONFIG, failed)
            @test DS.is_feasible_successor(blocked, r, nominal, action, r, nominal)
            @test length(DS.feasible_successors(blocked, r, nominal, action)) == 14
            key = DS.CountKey(blocked, r, nominal, action, DS.UNSAFE_CONFIG, nominal)
            @test DS.count_value(blocked, DS.increment_count(blocked, DS.DirichletCounts(), key), key) == 2
        end
        for (action, position) in enumerate(((3, 5), (3, 1), (1, 3), (5, 3)))
            source = DS.DSState(position, 0)
            failed = ntuple(i -> i == action, 6)
            @test !DS.is_feasible_successor(model, source, nominal, action, DS.UNSAFE_CONFIG, nominal)
            @test !DS.is_feasible_successor(model, source, nominal, action, source, failed)
            @test !DS.is_feasible_successor(model, source, nominal, action, DS.UNSAFE_CONFIG, failed)
            @test !DS.is_feasible_successor(model, source, nominal, action, source, nominal)
            @test_throws ArgumentError DS.feasible_successors(model, source, nominal, action)
        end
        source = DS.DSState((4, 5), 0)
        goal = DS.DSState(model.region_B, 0)
        @test DS.is_feasible_successor(model, source, nominal, 4, goal, nominal)
        @test DS.is_goal(DS.adaptive_state(model, goal))
        for terminal in (goal, DS.UNSAFE_CONFIG), mode in modes, action in 1:6
            @test DS.feasible_successors(model, terminal, mode, action) == [(terminal, mode)]
            @test DS.is_feasible_successor(model, terminal, mode, action, terminal, mode)
            @test_throws ArgumentError DS.count_row(model, DS.DirichletCounts(), terminal, mode, action)
            @test_throws ArgumentError DS.CountKey(model, terminal, mode, action, terminal, mode)
        end
        @test !DS.is_feasible_successor(model, goal, nominal, 4, goal, modes[2])
    end

    @testset "Validation and automatic HDR rows" begin
        r = DS.DSState((3, 3), 0)
        for action in (0, 7)
            @test_throws ArgumentError DS.feasible_successors(model, r, nominal, action)
            @test !DS.is_feasible_successor(model, r, nominal, action, r, nominal)
        end
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2)])
        for bad_source in (DS.DSState((0, 1)), DS.DSState((100, 100)), DS.DSState((2, 2)))
            @test_throws ArgumentError DS.feasible_successors(blocked, bad_source, nominal, 4)
            @test !DS.is_feasible_successor(blocked, bad_source, nominal, 4, r, nominal)
        end
        phi = DS.DirichletCounts()
        for action in 1:6
            support = DS.feasible_successors(model, r, nominal, action)
            rp, mp = support[1]
            learned = DS.increment_count(model, phi, DS.CountKey(model, r, nominal, action, rp, mp))
            state = DS.adaptive_state(model, r, nominal, learned)
            hdr = DS.build_row_hdr(model, state, action; epsilon=0.1, n_samples=100,
                                   rng=MersenneTwister(action))
            @test hdr.successors == support
            @test hdr.distribution.alpha == vcat([2], ones(Int, 13))
        end
        all_failed = last(modes)
        state = DS.adaptive_state(model, r, all_failed)
        hdr = DS.build_row_hdr(model, state, 4; epsilon=0.1, n_samples=1)
        @test hdr.successors == [(r, all_failed), (DS.DSState((4, 3), 0), all_failed)]
        @test hdr.distribution.alpha == [1, 1]
        @test hdr.tau == 1.0
        @test [0.5, 0.5] in hdr
        for terminal in (DS.adaptive_state(model, DS.UNSAFE_CONFIG),
                         DS.adaptive_state(model, DS.DSState(model.region_B)))
            @test_throws ArgumentError DS.build_row_hdr(model, terminal, 4; epsilon=0.1)
        end
        mislabeled = DS.AdaptiveState(r, nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.build_row_hdr(model, mislabeled, 4; epsilon=0.1)
    end
end
