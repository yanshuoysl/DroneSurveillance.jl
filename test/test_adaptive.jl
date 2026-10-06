using Test
using Random
using POMDPs
using POMDPTools

include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
include("adaptive_demo_route.jl")

@testset "DroneSurveillanceAdaptive" begin
    DS = DroneSurveillanceAdaptive
    rng = MersenneTwister(42)
    pomdp = DS.DroneSurveillancePOMDP(size=(3, 4), region_A=[2, 2], obstacles=DS.DSPos[])
    s0 = rand(rng, initialstate(pomdp))
    @test s0 == DS.DSState([2, 2])
    @test fieldnames(DS.DSState) == (:quad, :heading)
    @test s0.heading == 0
    @test rand(rng, initialstate(DS.DroneSurveillancePOMDP(initial_heading=90))).heading == 90
    @test_throws ArgumentError DS.DSState([2, 2], 45)
    @test length(pomdp) == 49
    @test length(DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])) == 101
    @test length(DS.DroneSurveillancePOMDP()) == 93
    @test ordered_states(pomdp) == collect(states(pomdp))
    for (i, s) in enumerate(states(pomdp))
        @test stateindex(pomdp, s) == i
    end

    @test collect(actions(pomdp, s0)) == [1, 2, 3, 4, 5, 6]
    @test ordered_actions(pomdp) == collect(actions(pomdp))
    @test DS.ACTIONS_DICT == Dict(:forward => 1, :backward => 2, :move_left => 3,
                                 :move_right => 4, :rotate_clockwise => 5,
                                 :rotate_counterclockwise => 6)
    # Expected absolute movement for each heading, in action order (1:4).
    moves = (
        (0, ([2, 3], [2, 1], [1, 2], [3, 2])),
        (90, ([3, 2], [1, 2], [2, 3], [2, 1])),
        (180, ([2, 1], [2, 3], [3, 2], [1, 2])),
        (270, ([1, 2], [3, 2], [2, 1], [2, 3])),
    )
    for (heading, positions) in moves
        s = DS.DSState([2, 2], heading)
        for (a, pos) in enumerate(positions)
            @test rand(rng, transition(pomdp, s, a)) == DS.DSState(pos, heading)
            @test actionindex(pomdp, a) == a
        end
        cw = rand(rng, transition(pomdp, s, 5))
        ccw = rand(rng, transition(pomdp, s, 6))
        @test cw == DS.DSState(s.quad, mod(heading + 90, 360))
        @test ccw == DS.DSState(s.quad, mod(heading - 90, 360))
        @test rand(rng, transition(pomdp, cw, 6)) == s
        @test rand(rng, transition(pomdp, ccw, 5)) == s
        for a in (5, 6)
            rotated = s
            for _ in 1:4
                rotated = rand(rng, transition(pomdp, rotated, a))
            end
            @test rotated == s
        end
        @test render(pomdp, (s=s,)) !== nothing
    end
    for (pos, heading, a) in (([1, 1], 0, 2), ([1, 1], 0, 3),
                              ([3, 4], 0, 1), ([3, 4], 90, 1),
                              ([1, 1], 180, 1), ([1, 1], 270, 1))
        @test rand(rng, transition(pomdp, DS.DSState(pos, heading), a)) == pomdp.terminal_state
    end
    @test !isterminal(pomdp, rand(rng, transition(pomdp, DS.DSState([1, 1]), 5)))
    @test !isterminal(pomdp, rand(rng, transition(pomdp, DS.DSState([3, 4]), 6)))
    @test_throws ArgumentError transition(pomdp, s0, 0)
    @test_throws ArgumentError transition(pomdp, s0, 7)
    for a in actions(pomdp)
        @test rand(rng, transition(pomdp, pomdp.terminal_state, a)) == pomdp.terminal_state
    end
    @test reward(pomdp, s0, 5) == 0.0
    @test reward(pomdp, DS.DSState(pomdp.region_B), 5) == 1.0
    @test reward(pomdp, pomdp.terminal_state, 5) == 0.0
    @test discount(pomdp) == 0.95
    @test has_consistent_transition_distributions(pomdp)
    @test has_consistent_observation_distributions(pomdp)

    @testset "Front-facing stochastic camera" begin
        camera_pomdp = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        @test length(observations(camera_pomdp)) == 65
        @test ordered_observations(camera_pomdp) == collect(observations(camera_pomdp))
        for (i, o) in enumerate(observations(camera_pomdp))
            @test obsindex(camera_pomdp, o) == i
        end
        @test_throws ArgumentError obsindex(camera_pomdp, DS.DSObservation(0, 1, 1))

        # Check ordered FOV coordinates against manually specified geometry.
        expected_cells = (
            (0, ([2, 4], [3, 4], [4, 4])),
            (90, ([4, 4], [4, 3], [4, 2])),
            (180, ([4, 2], [3, 2], [2, 2])),
            (270, ([2, 2], [2, 3], [2, 4])),
        )
        for (heading, positions) in expected_cells
            s = DS.DSState([3, 3], heading)
            @test DS.visible_cells(camera_pomdp, s) == map(DS.DSPos, positions)
            @test !DS.in_fov(camera_pomdp, s, s.quad)
            @test !DS.in_fov(camera_pomdp, s, DS.DSPos(3, 3) - DS.HEADING_DIRS[div(heading, 90) + 1])
            @test count(pos -> DS.in_fov(camera_pomdp, s, pos),
                        (DS.DSPos(x, y) for x in 1:5, y in 1:5)) == 3
        end
        @test DS.visible_cells(camera_pomdp, camera_pomdp.terminal_state) == ()

        s = DS.DSState([4, 4], 0) # ordinary, ordinary, survey ahead
        d = observation(camera_pomdp, 1, s)
        correct = DS.DSObservation(1, 1, 2)
        @test pdf(d, correct) ≈ 0.9^3
        @test pdf(d, DS.DSObservation(2, 1, 2)) ≈ (0.1/3) * 0.9^2
        @test sum(pdf(d, o) for o in observations(camera_pomdp)) ≈ 1.0
        @test pdf(d, DS.TERMINAL_OBSERVATION) == 0.0
        @test DS.observation_labels(correct) == (:ordinary, :ordinary, :survey)
        @test obstype(camera_pomdp) == DS.DSObservation
        samples = [rand(rng, d) for _ in 1:3000]
        @test length(unique(samples)) > 1
        @test all(o -> !(o isa DS.DSState) && length(o) == 3, samples)
        @test count(o -> o[1] == 1, samples)/length(samples) ≈ 0.9 atol=0.03
        @test count(o -> o[3] == 2, samples)/length(samples) ≈ 0.9 atol=0.03

        perfect = DS.DroneSurveillancePOMDP(camera=DS.KnownCamera(correct_probability=1.0), obstacles=DS.DSPos[])
        @test rand(rng, observation(perfect, 1, s)) == correct
        @test pdf(observation(perfect, 1, s), correct) == 1.0
        @test rand(rng, observation(perfect, 1, DS.DSState([1, 1]))) == DS.DSObservation(3, 1, 1)
        @test rand(rng, observation(perfect, 1, DS.DSState([1, 5]))) == DS.DSObservation(3, 3, 3)
        @test has_consistent_observation_distributions(perfect)

        # An asymmetric model verifies the true-class row / reported-class column convention.
        custom = DS.KnownCamera([0.8 0.15 0.03 0.02; 0.1 0.7 0.15 0.05;
                                0.2 0.1 0.6 0.1; 0.1 0.1 0.1 0.7])
        custom_pomdp = DS.DroneSurveillancePOMDP(camera=custom, obstacles=DS.DSPos[])
        @test pdf(observation(custom_pomdp, 1, s), correct) ≈ 0.8^2 * 0.7
        @test pdf(observation(custom_pomdp, 1, s), DS.DSObservation(2, 1, 3)) ≈ 0.15 * 0.8 * 0.15
        @test has_consistent_observation_distributions(custom_pomdp)
        for p in (-0.1, 1.1, NaN, Inf)
            @test_throws ArgumentError DS.KnownCamera(correct_probability=p)
        end
        @test_throws ArgumentError DS.KnownCamera(ones(2, 2))
        @test_throws ArgumentError DS.KnownCamera(zeros(3, 3))
        @test_throws ArgumentError DS.KnownCamera(zeros(4, 4))
        @test_throws ArgumentError DS.KnownCamera([1.1 -0.1 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
        @test_throws ArgumentError DS.KnownCamera([NaN 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])

        terminal_d = observation(camera_pomdp, 1, camera_pomdp.terminal_state)
        @test rand(rng, terminal_d) == DS.TERMINAL_OBSERVATION
        @test pdf(terminal_d, DS.TERMINAL_OBSERVATION) == 1.0
        @test DS.observation_labels(DS.TERMINAL_OBSERVATION) == (:terminal, :terminal, :terminal)

        # Verify the new observation type works with a Bayesian belief updater.
        up = DiscreteUpdater(camera_pomdp)
        b = initialize_belief(up, initialstate(camera_pomdp))
        turned = DS.DSState([1, 1], 90)
        o = rand(rng, observation(camera_pomdp, 5, turned))
        bp = update(up, b, 5, o)
        @test pdf(bp, turned) ≈ 1.0
    end

    @testset "Obstacles" begin
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2), DS.DSPos(3, 2)])
        @test length(blocked) == 93
        @test all(s -> isterminal(blocked, s) || s.quad ∉ blocked.obstacles, states(blocked))
        for (i, s) in enumerate(states(blocked))
            @test stateindex(blocked, s) == i
        end
        @test_throws ArgumentError stateindex(blocked, DS.DSState([2, 2]))
        @test_throws BoundsError DS.state_from_index(blocked, 0)
        @test_throws BoundsError DS.state_from_index(blocked, length(blocked) + 1)
        # A blocked translation keeps both position and heading for all action types.
        for (pos, heading, a) in (([2, 1], 0, 1), ([2, 3], 0, 2),
                                  ([3, 2], 0, 3), ([1, 2], 0, 4),
                                  ([1, 2], 90, 1))
            # Use an obstacle-free source cell for the left-motion case.
            local_model = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2)])
            s = DS.DSState(pos, heading)
            @test rand(rng, transition(local_model, s, a)) == s
        end
        @test has_consistent_transition_distributions(blocked)
        @test has_consistent_observation_distributions(blocked)
        @test render(blocked, (s=DS.DSState([2, 1]),)) !== nothing
        perfect = DS.DroneSurveillancePOMDP(obstacles=blocked.obstacles,
                                           camera=DS.KnownCamera(correct_probability=1.0))
        o = rand(rng, observation(perfect, 1, DS.DSState([2, 1])))
        @test o == DS.DSObservation(1, 4, 4)
        @test DS.observation_labels(o) == (:ordinary, :obstacle, :obstacle)
        noisy = observation(blocked, 1, DS.DSState([2, 1]))
        @test pdf(noisy, o) ≈ 0.9^3
        @test pdf(noisy, DS.DSObservation(1, 1, 4)) ≈ 0.9^2 * (0.1/3)

        first_map = DS.random_obstacles((5, 5); exclude=[DS.DSPos(1, 1), DS.DSPos(5, 5)], rng=MersenneTwister(7))
        @test first_map == DS.random_obstacles((5, 5); exclude=[DS.DSPos(1, 1), DS.DSPos(5, 5)], rng=MersenneTwister(7))
        @test isempty(DS.random_obstacles((1, 1); count=0))
        @test_throws ArgumentError DS.random_obstacles((1, 1); count=2)
        @test_throws ArgumentError DS.random_obstacles((0, 5))
        @test_throws AssertionError DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(1, 1)])
        @test_throws AssertionError DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(5, 5)])
        @test_throws AssertionError DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(6, 1)])
        @test_throws AssertionError DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 2), DS.DSPos(2, 2)])

        for seed in 1:20
            waypoint = DS.DSPos(3, 3)
            model, route = random_demo_environment(MersenneTwister(seed), waypoint)
            @test length(unique(model.obstacles)) == 2
            @test all(pos -> 1 <= pos[1] <= 5 && 1 <= pos[2] <= 5, model.obstacles)
            @test all(pos -> pos ∉ model.obstacles, [model.region_A, waypoint, model.region_B])
            s = rand(rng, initialstate(model))
            visited = DS.DSPos[s.quad]
            for a in route
                s = rand(rng, transition(model, s, a))
                push!(visited, s.quad)
            end
            @test all(pos -> pos ∉ model.obstacles, visited)
            @test waypoint in visited
            @test visited[end] == model.region_B
        end
        sealed = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 1), DS.DSPos(1, 2)])
        @test demo_waypoint_route(sealed, [sealed.region_B]) === nothing
        detour = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(2, 1), DS.DSPos(3, 2)])
        path = demo_grid_path(detour, detour.region_A, DS.DSPos(3, 3))
        @test path !== nothing
        @test path[2] == DS.DSPos(1, 2)
        @test all(pos -> pos ∉ detour.obstacles, path)
    end

    # Turn east, move forward, turn north, reach B, then rotate in place.
    s = s0
    for a in (5, 1, 6, 1, 1, 5)
        s = rand(rng, transition(pomdp, s, a))
        @test !isterminal(pomdp, s)
    end
    @test s.quad == pomdp.region_B
    policy = RandomPolicy(pomdp, rng=rng)
    hist = simulate(HistoryRecorder(max_steps=10, rng=rng), pomdp, policy)
    @test length(hist) > 0
    @test render(pomdp, (s=s0,)) !== nothing
    @test render(pomdp, (s=pomdp.terminal_state,)) !== nothing
    @test render(pomdp) !== nothing
end
