using Test
using Random
using POMDPs
using POMDPTools

if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

@testset "Paper-layout robust finite-horizon planning" begin
    DS = DroneSurveillanceAdaptive
    all_faults = ntuple(_ -> true,6)
    options = DS.RobustPlannerOptions(gap_tolerance=2e-6)
    vf_for(p,n; kwargs...) = DS.RobustValueFunction(p;horizon=n,epsilon=0.1,n_samples=64,options,kwargs...)

    function paper_value(p,b,result,next_vf,n)
        # Independently enumerate the full transition and the real observations;
        # use the pure exact updater to evaluate each hypothetical posterior.
        reward = 0.0
        evidence = Dict{DS.DSObservation,Float64}()
        for (s,w) in b
            distribution = if DS.is_terminal(s)
                DS.adaptive_transition(p,s,result.action)
            else
                row = result.selected_rows._rows[DS.freeze_hyperstate(s)]
                hdr = DS.DirichletHDR(DS.hdr_distribution(collect(row.alpha)),collect(row.successors),
                                     row.epsilon,row.n_samples,row.tau)
                DS.robust_transition(p,s,result.action,collect(row.theta);hdr)
            end
            for (t,probability) in weighted_iterator(distribution)
                reward += w*probability*DS.adaptive_reward(p,s,result.action,t)
                for (o,likelihood) in weighted_iterator(observation(p,result.action,t))
                    likelihood == 0 && continue
                    evidence[o] = get(evidence,o,0.0) + w*probability*likelihood
                end
            end
        end
        future = 0.0
        for (o,probability) in evidence
            probability == 0 && continue
            posterior = DS.exact_belief_update(p,b,result.action,o,result.selected_rows)
            future += probability*value(next_vf,posterior,n+1)
        end
        return reward, future, reward+discount(p)*future
    end

    @testset "One-step reward, action maximum, and counts" begin
        p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(3,5)])
        s = DS.adaptive_state(p,DS.DSState((4,5)))
        b = DS.ExactAdaptiveBelief(p,s)
        vf = vf_for(p,1)
        before = deepcopy(s)
        decision = DS.robust_decision(p,b,0,vf)
        @test decision.action == 4 && decision.status == :converged
        @test [r.action for r in decision.action_results] == actions(p,b)
        @test decision.selection_gap <= options.gap_tolerance
        @test decision.value == maximum(r.lower_bound for r in decision.action_results)
        @test value(vf,b,0) ≈ decision.value atol=options.gap_tolerance
        for result in decision.action_results
            @test result.gap <= options.gap_tolerance/2
            @test result.lower_bound <= result.upper_bound
            @test result.value == result.lower_bound
            reward,future,total = paper_value(p,b,result,vf,0)
            @test result.current_reward ≈ reward
            @test future == result.expected_future_value == 0
            @test result.upper_bound ≈ total
        end
        goal = only(filter(r->r.action==4,decision.action_results))
        collision = only(filter(r->r.action==3,decision.action_results))
        @test goal.value > -1 && collision.value < -1
        @test s == before && isempty(s.counts.increments)
        @test length(sprint(show,vf)) < 200
        @test length(sprint(show,decision)) < 250
        @test all(length(sprint(show,r)) < 250 for r in decision.action_results)

        up = DS.ExactAdaptiveUpdater(p)
        @test_throws ArgumentError update(up,b,decision.action,DS.TERMINAL_OBSERVATION)
        DS.prepare_decision!(up,b,decision)
        posterior = update(up,b,decision.action,DS.TERMINAL_OBSERVATION)
        @test length(posterior) == 7 && all(DS.is_goal(t) for t in support(posterior))
        @test all(sum(values(t.counts.increments)) == 1 for t in support(posterior))
        @test_throws ArgumentError update(up,b,decision.action,DS.TERMINAL_OBSERVATION)
        changed = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((4,4))))
        @test_throws ArgumentError DS.prepare_decision!(up,changed,decision)
    end

    @testset "Belief weighting preserves distinct count histories" begin
        p = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        s1 = DS.adaptive_state(p,DS.DSState((4,5)),all_faults)
        stay = DS.CountKey(p,s1.config,all_faults,4,s1.config,all_faults)
        s2 = DS.adaptive_state(p,s1.config,all_faults,DS.DirichletCounts(Dict(stay=>3)))
        b = DS.ExactAdaptiveBelief(p,[s1=>0.3,s2=>0.7])
        vf = vf_for(p,1)
        result = DS.backup_action(p,b,4,0,vf)
        expected = 0.0
        for (s,w) in b
            hdr = DS.build_row_hdr(p,s,4;epsilon=0.1,n_samples=64,rng=MersenneTwister(42))
            coefficients = [DS.adaptive_reward(p,s,4,DS.adaptive_state(p,rp,mp)) for (rp,mp) in hdr.successors]
            expected += w*DS.minimize_hdr_row(hdr,coefficients).value
        end
        @test result.upper_bound ≈ expected atol=options.gap_tolerance
        @test length(result.continuation.support) == 2
        @test length(result.selected_rows._rows) == 2
        @test isempty(s1.counts.increments) && s2.counts.increments == Dict(stay=>3)
        @test result.continuation.coefficients[1] != result.continuation.coefficients[2]
        terminal = DS.adaptive_state(p,DS.DSState(p.region_B),all_faults)
        mixed = DS.ExactAdaptiveBelief(p,[s1=>0.3,terminal=>0.7])
        mixed_result = DS.backup_action(p,mixed,4,0,vf)
        active_result = DS.backup_action(p,DS.ExactAdaptiveBelief(p,s1),4,0,vf)
        @test mixed_result.upper_bound ≈ 0.3active_result.upper_bound
        @test length(mixed_result.selected_rows._rows) == 1
        @test isempty(terminal.counts.increments)
    end

    @testset "Policy adapter prepares before observing and advances only on success" begin
        p = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        b = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((4,5)),all_faults))
        vf = vf_for(p,1)
        policy = DS.RobustPolicy(vf)
        up = updater(policy)
        @test_throws ArgumentError update(up,b,4,DS.TERMINAL_OBSERVATION)
        a = action(policy,b)
        @test a == 4 && policy.stage == 0
        pending = policy._pending
        @test action(policy,b) == a && policy._pending === pending
        @test_throws ArgumentError update(up,b,2,DS.TERMINAL_OBSERVATION)
        @test_throws ArgumentError update(up,b,a,DS.DSObservation(5,1,1))
        @test policy.stage == 0 && policy._pending === pending
        policy.stage = 1
        @test_throws ArgumentError action(policy,b)
        @test_throws ArgumentError update(up,b,a,DS.TERMINAL_OBSERVATION)
        policy.stage = 0
        posterior = update(up,b,a,DS.TERMINAL_OBSERVATION)
        @test policy.stage == 1 && policy._pending === nothing
        @test all(DS.is_goal(s) for s in support(posterior))
        @test all(sum(values(s.counts.increments)) == 1 for s in support(posterior))
        @test_throws ArgumentError action(policy,posterior)
        @test_throws ArgumentError update(up,b,a,DS.TERMINAL_OBSERVATION)
        DS.reset_robust_policy!(policy)
        @test policy.stage == 0 && policy._exact_updater._pending === nothing
        action(policy,b)
        initialized = initialize_belief(up,b)
        @test policy.stage == 0 && policy._pending === nothing
        @test pdf(initialized,only(support(b))) == 1
        @test length(sprint(show,policy)) < 200
        @test_throws ArgumentError DS.reset_robust_policy!(policy;stage=2)
    end

    @testset "Two-step observation-dependent actions and independent dense search" begin
        results = Float64[]
        for camera in (DS.KnownCamera(correct_probability=1),DS.KnownCamera(),DS.KnownCamera(fill(0.25,4,4)))
            p = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(3,3),obstacles=DS.DSPos[],camera=camera)
            s = DS.adaptive_state(p,DS.DSState((3,2)),all_faults)
            b = DS.ExactAdaptiveBelief(p,s)
            vf = vf_for(p,2)
            # Rotating clockwise leaves a north/east heading ambiguity. At the
            # next decision, north reaches the goal with forward, east with left.
            result = DS.backup_action(p,b,5,0,vf)
            reward,future,total = paper_value(p,b,result,vf,0)
            @test reward ≈ -1
            @test result.current_reward ≈ reward
            @test result.expected_future_value ≈ future
            @test result.upper_bound ≈ total atol=options.gap_tolerance
            @test result.gap <= options.gap_tolerance/2
            row = only(values(result.selected_rows._rows))
            @test sum(row.theta) ≈ 1
            @test row.theta[1] ≈ 0.5 atol=2e-5
            @test isempty(s.counts.increments)

            # Independent one-dimensional reference. Future goal rows have the
            # untouched Dir(2,2) prior, irrespective of which rotation occurred.
            goal_hdr = DS.build_row_hdr(p,s,1;epsilon=0.1,n_samples=64,rng=MersenneTwister(42))
            low = (goal_hdr.tau/3)/(1+sqrt(1-2goal_hdr.tau/3))
            goal_value = -1+101low
            north = DS.adaptive_state(p,s.config,all_faults)
            east = DS.adaptive_state(p,DS.DSState(s.config.quad,90),all_faults)
            observations = collect(POMDPs.observations(p))
            ln = [pdf(observation(p,5,north),o) for o in observations]
            le = [pdf(observation(p,5,east),o) for o in observations]
            function independent(q)
                success = sum(max.((1-q).*ln,q.*le))
                return -1+discount(p)*(-1+(goal_value+1)*success)
            end
            grid_minimum = minimum(independent(q) for q in range(low,1-low;length=2001))
            @test result.lower_bound <= grid_minimum + 2e-7
            @test result.upper_bound ≈ grid_minimum atol=3e-6
            push!(results,result.value)
            if camera.confusion[1,1] == 0.9
                @test result.iterations > 1 # real continuation/model coupling
            end
        end
        @test results[1] > results[2] > results[3]
        p = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(3,3),obstacles=DS.DSPos[])
        b = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((3,2)),all_faults))
        limited = vf_for(p,2;options=DS.RobustPlannerOptions(max_iterations=1))
        @test_throws DS.RobustPlanningError DS.backup_action(p,b,5,0,limited)
    end

    @testset "Final continuation rows do not cache successor histories" begin
        p = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(3,3),obstacles=DS.DSPos[])
        source = DS.adaptive_state(p,DS.DSState((3,2)),all_faults)
        b = DS.ExactAdaptiveBelief(p,source)
        # The root has six actions with two edges each. That is the complete
        # edge budget: final-layer row minima must not cache grandchildren.
        limited = vf_for(p,2;options=DS.RobustPlannerOptions(
            max_total_edges=12,max_edges=2,max_cached_actions=6))
        reference = vf_for(p,2)
        for vf in (limited,reference)
            DS.fit_value_function!(vf,b)
        end
        decision = DS.robust_decision(p,b,0,limited)
        expected = DS.robust_decision(p,b,0,reference)
        @test decision.action == expected.action
        @test [r.lower_bound for r in decision.action_results] ≈
              [r.lower_bound for r in expected.action_results]
        @test [r.upper_bound for r in decision.action_results] ≈
              [r.upper_bound for r in expected.action_results]
        @test decision.selected_rows._rows == expected.selected_rows._rows
        @test length(limited._actions) == 6
        @test sum(length(a.children) for a in values(limited._actions)) == 12
        @test all(observations for ((_,_,observations),_) in limited._actions)
        @test isempty(source.counts.increments)

        # Exercise the large supports seen at L=2. Cache lookup types must not
        # grow with the number of hypotheses, and equal snapshots must reuse
        # the same completed family.
        historical = DS.CountKey(p,DS.DSState((4,4)),all_faults,1,DS.DSState((4,4)),all_faults)
        large = Tuple(DS.freeze_hyperstate(DS.adaptive_state(p,source.config,all_faults,
                          DS.DirichletCounts(Dict(historical=>i)))) for i in 1:1024)
        large_vf = DS.RobustValueFunction(p;horizon=2,epsilon=.1,n_samples=16,
            options=DS.RobustPlannerOptions(max_total_edges=1))
        family = DS.continuation_family!(large_vf,large,1)
        @test isempty(large_vf._actions)
        @test length(large_vf._hdrs) == 6*1024
        @test all(length(a.coefficients) == 1024 for a in family)
        @test DS.continuation_family!(large_vf,Tuple(collect(large)),1) === family
        @test length(large_vf._families) == length(large_vf._last_complete) == 1

        # Compare the streamed coefficients to the full transition expansion,
        # including a terminal component and a nontrivial learned count row.
        stay = DS.CountKey(p,source.config,all_faults,1,source.config,all_faults)
        learned = DS.adaptive_state(p,source.config,all_faults,DS.DirichletCounts(Dict(stay=>3)))
        terminal = DS.adaptive_state(p,DS.DSState(p.region_B),all_faults)
        mixed = DS.ExactAdaptiveBelief(p,[source=>0.2,learned=>0.3,terminal=>0.5])
        vf = vf_for(p,2)
        states = DS.planning_support(vf,mixed)
        for a in actions(p,mixed)
            expanded = DS.prepare_robust_action!(vf,states,a;need_observations=false)
            for (i,parent) in enumerate(states)
                streamed = DS.planner_immediate_minimum!(vf,parent,a)
                result = DS.planner_row_minimum(vf,expanded,i,expanded.rewards[expanded.ranges[i]])
                if result === nothing
                    @test streamed === nothing
                else
                    # Independently solve the expanded row, bypassing the shared cache.
                    direct = DS.minimize_hdr_row(expanded.hdrs[i],expanded.rewards[expanded.ranges[i]];
                        log_atol=1e-8,objective_atol=min(1e-10,vf.options.gap_tolerance*1e-3),objective_rtol=1e-12)
                    @test streamed.objective_lower_bound ≈ direct.objective_lower_bound
                    @test streamed.theta ≈ direct.theta
                end
            end
        end
    end

    @testset "Backward value fitting, explicit stage, and updater isolation" begin
        p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(3,5)],camera=DS.KnownCamera(correct_probability=1))
        b = DS.ExactAdaptiveBelief(p,DS.adaptive_state(p,DS.DSState((4,5)),all_faults))
        vf = vf_for(p,3)
        before = value(vf,b,0)
        @test before ≈ -(1+discount(p)+discount(p)^2)
        report = DS.fit_value_function!(vf,b;expand=true)
        @test report.points > 1 && report.updates > 0
        @test report.approximation == :finite_continuations
        @test value(vf,b,0) >= before
        @test all(n in 0:3 for ((n,_),_) in vf._families)
        @test any(n==1 for ((n,_),_) in vf._families)
        @test any(n==2 for ((n,_),_) in vf._families)
        @test value(vf,b,3) == 0
        decision = DS.robust_decision(p,b,0,vf)
        @test decision.action in actions(p,b)
        @test decision.selection_gap <= options.gap_tolerance
        up = DS.ExactAdaptiveUpdater(p)
        DS.prepare_decision!(up,b,decision)
        pending = up._pending
        second = DS.fit_value_function!(vf,Dict(0=>[b]);sweeps=1)
        @test second.points == 1 && up._pending === pending
        @test isempty(only(support(b)).counts.increments)
    end

    @testset "Terminal, horizon, boundaries, calibration, and explicit limits" begin
        p = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[])
        b = DS.ExactAdaptiveBelief(p)
        vf0 = vf_for(p,0)
        @test value(vf0,b,0) == 0
        horizon = DS.robust_decision(p,b,0,vf0)
        @test horizon.action === nothing && horizon.status == :horizon
        @test_throws ArgumentError DS.prepare_decision!(DS.ExactAdaptiveUpdater(p),b,horizon)
        @test_throws ArgumentError DS.backup_action(p,b,1,0,vf0)
        vf = vf_for(p,1)
        for s in (DS.adaptive_state(p,DS.DSState(p.region_B)),DS.adaptive_state(p,DS.UNSAFE_CONFIG))
            terminal = DS.ExactAdaptiveBelief(p,s)
            @test value(vf,terminal,0) == 0
            decision = DS.robust_decision(p,terminal,0,vf)
            @test decision.action === nothing && decision.status == :terminal
            @test isempty(decision.action_results)
        end
        rare = DS.adaptive_state(p,DS.DSState((1,3)))
        common = DS.adaptive_state(p,DS.DSState((3,3)))
        uncertain = DS.ExactAdaptiveBelief(p,[rare=>-1000.0,common=>0.0];logweights=true)
        decision = DS.robust_decision(p,uncertain,0,vf)
        @test [r.action for r in decision.action_results] == collect(1:6)
        @test length(decision.selected_rows._rows) == 2
        @test isfinite(DS.backup_action(p,uncertain,3,0,vf).value)
        boundary_risk = DS.ExactAdaptiveBelief(p,[rare=>.1,common=>.9])
        @test DS.backup_action(p,boundary_risk,3,0,vf).value < -1
        @test_throws ArgumentError value(vf,b,-1)
        @test_throws ArgumentError value(vf,b,2)
        @test_throws ArgumentError DS.fit_value_function!(vf,Dict(0=>b))
        @test_throws ArgumentError DS.fit_value_function!(vf,b;sweeps=0)
        @test_throws ArgumentError DS.robust_decision(DS.DroneSurveillancePOMDP(),b,0,vf)

        # Cache order does not alter calibration; model/calibration edits fail.
        vf1,vf2 = vf_for(p,1),vf_for(p,1)
        r1 = DS.backup_action(p,b,1,0,vf1)
        DS.backup_action(p,b,4,0,vf2)
        r2 = DS.backup_action(p,b,1,0,vf2)
        @test r1.value ≈ r2.value && r1.selected_rows._rows == r2.selected_rows._rows
        @test length(vf1._hdrs) == 1
        vf1.epsilon = 0.2
        @test_throws DS.RobustPlanningError value(vf1,b,0)
        old = p.discount_factor
        p.discount_factor = 0.5
        @test_throws DS.RobustPlanningError DS.robust_decision(p,b,0,vf2)
        p.discount_factor = old

        for opts in (DS.RobustPlannerOptions(max_hypotheses=1),
                     DS.RobustPlannerOptions(max_edges=1),
                     DS.RobustPlannerOptions(max_total_edges=1),
                     DS.RobustPlannerOptions(max_score_entries=1))
            limited = vf_for(p,2;options=opts)
            @test_throws DS.RobustPlanningError DS.backup_action(p,b,1,0,limited)
        end
        limited = vf_for(p,3;options=DS.RobustPlannerOptions(max_training_points=1))
        @test_throws DS.RobustPlanningError DS.fit_value_function!(limited,b)
        @test_throws ArgumentError vf_for(p,1;options=DS.RobustPlannerOptions(gap_tolerance=-1.0))
        @test_throws ArgumentError vf_for(p,1;options=DS.RobustPlannerOptions(max_iterations=0))
        @test_throws ArgumentError DS.RobustValueFunction(p;horizon=-1,epsilon=0.1)

        # Observation likelihoods can underflow in numeric value calculations,
        # but structural support must retain them without restricting actions.
        confusion = fill(1e-200,4,4)
        for i in 1:4
            confusion[i,i] = 1
        end
        tiny = DS.DroneSurveillancePOMDP(obstacles=DS.DSPos[],camera=DS.KnownCamera(confusion))
        tiny_b = DS.ExactAdaptiveBelief(tiny,DS.adaptive_state(tiny,DS.DSState((3,3)),all_faults))
        tiny_vf = vf_for(tiny,2)
        states = DS.planning_support(tiny_vf,tiny_b)
        prepared = DS.prepare_robust_action!(tiny_vf,states,4)
        index = findfirst(==(DS.DSObservation(4,4,4)),prepared.observations)
        @test index !== nothing
        @test all(isfinite,prepared.log_likelihoods[:,index])
        @test all(==(0.0),exp.(prepared.log_likelihoods[:,index]))
        @test length(prepared.next_supports[index]) == 2

        coupled = DS.DroneSurveillancePOMDP(region_B=DS.DSPos(3,3),obstacles=DS.DSPos[])
        coupled_b = DS.ExactAdaptiveBelief(coupled,DS.adaptive_state(coupled,DS.DSState((3,2)),all_faults))
        limited = vf_for(coupled,2;options=DS.RobustPlannerOptions(max_vectors=1))
        @test_throws DS.RobustPlanningError DS.backup_action(coupled,coupled_b,5,0,limited)
    end
end
