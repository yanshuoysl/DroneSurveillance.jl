using Test
using Random
using POMDPs

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "AR transition reward" begin
    DS = DroneSurveillanceAdaptive
    nominal = DS.NOMINAL_MODE
    one_fault = (true, false, false, false, false, false)
    all_faults = (true, true, true, true, true, true)
    model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
    r = DS.DSState((3, 3), 0)

    @testset "Ordinary motion, stay, rotations, and uncertain modes cost one" begin
        for heading in (0, 90, 180, 270), mode in (nominal, one_fault, all_faults), action in 1:6
            s = DS.adaptive_state(model, DS.DSState((3, 3), heading), mode)
            for sp in DS.adaptive_transition(model, s, action).vals
                before = copy(s.counts.increments)
                next_before = copy(sp.counts.increments)
                @test DS.adaptive_reward(model, s, action, sp) == -1.0
                @test reward(model, s, action, sp) == -1.0
                @test s.counts.increments == before
                @test sp.counts.increments == next_before
            end
        end
    end

    @testset "Goal-entry reward is exactly 100, only on entry" begin
        p = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(3, 3), obstacles=DS.DSPos[])
        # Independent command offsets by heading; all four translations can
        # enter the goal, retaining their original heading.
        directions = (
            ((0, 1), (0, -1), (-1, 0), (1, 0)),
            ((1, 0), (-1, 0), (0, 1), (0, -1)),
            ((0, -1), (0, 1), (1, 0), (-1, 0)),
            ((-1, 0), (1, 0), (0, -1), (0, 1)),
        )
        for (hi, heading) in enumerate((0, 90, 180, 270)), action in 1:4
            source = DS.DSState(p.region_B - DS.DSPos(directions[hi][action]), heading)
            s = DS.adaptive_state(p, source)
            for mode in DS.successor_modes(nominal)
                sp = DS.adaptive_state(p, DS.DSState(p.region_B, heading), mode)
                @test DS.adaptive_reward(p, s, action, sp) == 100.0
                @test reward(p, s, action, sp) == 100.0
                for a in 1:6
                    @test DS.adaptive_reward(p, sp, a, sp) == 0.0
                end
            end
            stayed = DS.adaptive_state(p, source)
            @test DS.adaptive_reward(p, s, action, stayed) == -1.0
        end
    end

    @testset "Obstacle collisions cost 100; a stay before an obstacle costs one" begin
        targets = (
            ((3, 4), (3, 2), (2, 3), (4, 3)),
            ((4, 3), (2, 3), (3, 4), (3, 2)),
            ((3, 2), (3, 4), (4, 3), (2, 3)),
            ((2, 3), (4, 3), (3, 2), (3, 4)),
        )
        for (hi, heading) in enumerate((0, 90, 180, 270)), action in 1:4
            p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(targets[hi][action])])
            s = DS.adaptive_state(p, DS.DSState((3, 3), heading))
            for mode in DS.successor_modes(nominal)
                crashed = DS.adaptive_state(p, DS.UNSAFE_CONFIG, mode)
                stayed = DS.adaptive_state(p, s.config, mode)
                @test DS.adaptive_reward(p, s, action, crashed) == -100.0
                @test reward(p, s, action, crashed) == -100.0
                @test DS.adaptive_reward(p, s, action, stayed) == -1.0
                for a in 1:6
                    @test DS.adaptive_reward(p, crashed, a, crashed) == 0.0
                end
            end
        end
    end

    @testset "Outward boundary commands have no reward transition" begin
        for (action, position) in enumerate(((3, 5), (3, 1), (1, 3), (5, 3)))
            s = DS.adaptive_state(model, DS.DSState(position, 0))
            for mode in DS.successor_modes(nominal)
                outside = DS.adaptive_state(model, DS.UNSAFE_CONFIG, mode)
                @test_throws ArgumentError DS.adaptive_reward(model, s, action, outside)
                @test_throws ArgumentError DS.adaptive_reward(model, s, action, DS.adaptive_state(model, s.config, mode))
            end
        end
    end

    @testset "Full transition kernels carry the correct per-branch reward" begin
        goal_model = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        obstacle_model = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)])
        for (p, config, commanded_reward, mean_reward) in (
                (goal_model, DS.DSState((4, 5), 0), 100.0, 49.5),
                (obstacle_model, r, -100.0, -50.5))
            s = DS.adaptive_state(p, config)
            distribution = DS.adaptive_transition(p, s, 4)
            rewards = [DS.adaptive_reward(p, s, 4, sp) for sp in distribution.vals]
            @test rewards[1:7] == fill(-1.0, 7)
            @test rewards[8:14] == fill(commanded_reward, 7)
            @test sum(distribution.probs .* rewards) ≈ mean_reward
            # Changes to the parameter history do not affect the reward.
            key = DS.CountKey(p, config, nominal, 4, config, nominal)
            phi = DS.DirichletCounts(Dict(key => 99))
            learned_source = DS.adaptive_state(p, config, nominal, phi)
            learned_distribution = DS.adaptive_transition(p, learned_source, 4)
            @test [DS.adaptive_reward(p, learned_source, 4, sp) for sp in learned_distribution.vals] == rewards
            @test phi.increments == Dict(key => 99)
        end
    end

    @testset "Reject inconsistent or impossible transition inputs" begin
        s = DS.adaptive_state(model, r)
        right = DS.adaptive_state(model, DS.DSState((4, 3), 0))
        for action in (0, 7)
            @test_throws ArgumentError DS.adaptive_reward(model, s, action, right)
        end
        bad_source = DS.AdaptiveState(r, nominal; outcome=DS.GOAL_STATE)
        bad_target = DS.AdaptiveState(DS.DSState((4, 3), 0), nominal; outcome=DS.GOAL_STATE)
        @test_throws ArgumentError DS.adaptive_reward(model, bad_source, 4, right)
        @test_throws ArgumentError DS.adaptive_reward(model, s, 4, bad_target)
        @test_throws ArgumentError DS.adaptive_reward(model, s, 4,
                                                     DS.adaptive_state(model, DS.DSState((2, 3), 0)))
        @test_throws ArgumentError DS.adaptive_reward(model, DS.adaptive_state(model, r, one_fault), 4, right)
        blocked = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(4, 3)])
        raw_obstacle = DS.AdaptiveState(DS.DSState((4, 3), 0))
        @test_throws ArgumentError DS.adaptive_reward(blocked, s, 4, raw_obstacle)
        goal = DS.adaptive_state(model, DS.DSState(model.region_B))
        @test_throws ArgumentError DS.adaptive_reward(model, goal, 4, right)
    end
end
