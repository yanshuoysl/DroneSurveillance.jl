"""
    simulate_adaptive(pomdp, policy, belief_updater, initial_belief;
        H, environment=nothing, transition_kernel=nothing, initial_state=..., rng=...)

Execute until goal, collision, or H actions. Supply exactly one physical
`environment` or explicit custom `transition_kernel(p,s,a)` (useful for fixtures).
There is no implicit posterior-mean or adversarial physical model. Only this
runner/environment receives true states. Policy/updater inputs remain b and
(b,a,o). Custom kernels must produce complete hyperstates including counts.
History retains the terminal reward, observation, update, and truth diagnostics.
Phase timings exclude diagnostic logging and rendering. computation_seconds is
the sum of core phase wall times minus Julia-reported compilation (Julia >= 1.11);
GC remains included. Older Julia versions report this adjusted total as nothing.
For steady-state measurements, warm up the same workload using a separate policy.
"""
function simulate_adaptive(pomdp::DroneSurveillancePOMDP, policy, belief_updater,
        initial_belief; H::Integer, environment=nothing, transition_kernel=nothing,
        initial_state::AdaptiveState=initial_adaptive_state(pomdp),
        rng::AbstractRNG=Random.default_rng(), logger=nothing, reward_io=nothing)
    (environment === nothing) != (transition_kernel === nothing) ||
        throw(ArgumentError("supply exactly one environment or transition_kernel"))
    if environment !== nothing
        environment isa HiddenFaultEnvironment || throw(ArgumentError("invalid physical environment"))
        environment.H == H || throw(ArgumentError("physical and simulation horizons differ"))
        environment.initial_mode == initial_state.mode || throw(ArgumentError("initial physical modes differ"))
    end
    if policy isa AdaptiveRobustPolicy
        validate_episode_policy(pomdp,policy,H)
        (belief_updater isa RobustPolicyUpdater || belief_updater isa RecedingHorizonRobustUpdater) &&
            belief_updater.policy === policy || throw(ArgumentError("use the robust policy's matching updater"))
        initial_belief isa ExactAdaptiveBelief || throw(ArgumentError("robust policy needs an exact belief"))
    end
    s, b = initial_state, initial_belief
    steps, total_reward = 0, 0.0
    records = NamedTuple[]
    step_timings = NamedTuple[]
    reason = termination_reason(pomdp,s,steps,H)
    episode_event(logger,:start; H, initial_belief=logger === nothing ? nothing : belief_record(logger,b;pomdp),
        simulator_truth=(; state=configuration_record(s.config), mode=s.mode,
            fault_steps=environment === nothing ? nothing : environment.fault_steps,
            fault_actions=environment === nothing ? nothing : environment.fault_actions))
    if logger !== nothing && logger.terminal !== nothing
        println(logger.terminal,"AR episode: H=",H,"; true states/faults below are evaluation-only")
    end
    reward_io === nothing || (println(reward_io,"step,reward"); flush(reward_io))
    episode_start = time_ns()
    try
        while reason === nothing
            planning_stats = @timed Int64(action(policy,b))
            a = planning_stats.value
            a in actions(pomdp,s) || throw(ArgumentError("policy selected an unavailable action: $a"))
            decision = policy isa AdaptiveRobustPolicy ? pending_robust_decision(policy) : nothing
            episode_stage = policy isa AdaptiveRobustPolicy ? policy.stage : steps
            lookahead = policy isa AdaptiveRobustPolicy ? planning_window(policy) : nothing
            mode = policy isa AdaptiveRobustPolicy ? planning_mode(policy) : nothing
            fit = policy isa AdaptiveRobustPolicy ? policy_fit_summary(policy) : nothing
            log_decision!(logger,pomdp,b,a,decision,steps+1;
                episode_stage,lookahead,planning_mode=mode,fit)
            environment_stats = @timed begin
                physical = environment === nothing ? nothing : physical_step(pomdp,environment,s,a,steps+1)
                sp = physical === nothing ? rand(rng,transition_kernel(pomdp,s,a)) : physical.sp
                r = adaptive_reward(pomdp,s,a,sp)
                o = rand(rng,observation(pomdp,a,sp))
                (; physical, sp, r, o)
            end
            (;physical,sp,r,o) = environment_stats.value
            total_reward += r
            truth = (; source=configuration_record(s.config), source_mode=s.mode,
                source_available_actions=actions(pomdp,s),
                destination=configuration_record(sp.config), next_mode=sp.mode,
                destination_available_actions=actions(pomdp,sp),
                fault_event=physical === nothing ? nothing : physical.event,
                attempted=physical === nothing ? nothing : configuration_record(physical.attempted),
                count_key=physical === nothing ? nothing : count_key_record(physical.count_key),
                count_before=physical === nothing ? nothing : count_value(s.counts,physical.count_key),
                count_after=physical === nothing ? nothing : count_value(sp.counts,physical.count_key))
            episode_event(logger,:physical_transition; step=steps+1, observation=Tuple(o),
                observation_labels=observation_labels(o), reward=r, total_reward, simulator_truth=truth)
            reward_io === nothing || (println(reward_io,steps+1,',',r); flush(reward_io))
            if logger !== nothing && logger.terminal !== nothing
                println(logger.terminal,"  simulator truth BEFORE action (evaluation only): position=",Tuple(s.config.quad),
                    "; heading=",s.config.heading,"; available actions=",named_actions(truth.source_available_actions))
                println(logger.terminal,"  observation=",observation_labels(o),"; reward=",r,"; total=",total_reward)
                println(logger.terminal,"  simulator truth: position=",Tuple(sp.config.quad),
                    "; heading=",sp.config.heading,"; available actions=",named_actions(truth.destination_available_actions),
                    "; mode=",sp.mode,"; new fault=",truth.fault_event)
                physical === nothing || println(logger.terminal,"  realized phi key: ",truth.count_before,
                    " -> ",truth.count_after,"; source mode=",s.mode," -> ",sp.mode)
            end
            # Same pre-observation decision/rows. No physical state is supplied.
            belief_stats = @timed update(belief_updater,b,a,o)
            bp = belief_stats.value
            log_posterior!(logger,pomdp,b,bp,a,o,decision,steps+1;
                filter_report=policy isa AdaptiveRobustPolicy ? particle_filter_report(policy) : nothing)
            timing = (; planning=computation_phase_timing(planning_stats),
                environment=computation_phase_timing(environment_stats),
                belief_update=computation_phase_timing(belief_stats))
            push!(step_timings,timing)
            log_computation_timing!(logger,timing,steps+1)
            steps += 1
            push!(records,(; t=steps,s,a,sp,o,r,b,bp,timing,
                fault_event=truth.fault_event, attempted=truth.attempted,
                decision=decision === nothing ? nothing :
                    (; stage=decision.stage,episode_stage,lookahead,planning_mode=mode,
                       value=decision.value,selection_gap=decision.selection_gap,
                       action_values=[(;action=r.action,value=r.value) for r in decision.action_results])))
            s,b = sp,bp
            reason = termination_reason(pomdp,s,steps,H)
        end
    catch err
        episode_event(logger,:error; completed_steps=steps, message=sprint(showerror,err))
        logger === nothing || logger.terminal === nothing || println(logger.terminal,"Episode stopped on error: ",sprint(showerror,err))
        rethrow()
    end
    timing = computation_timing_summary(step_timings,(time_ns()-episode_start)/1e9)
    history = POMDPTools.SimHistory(records,discount(pomdp),nothing,nothing)
    episode_event(logger,:stop; reason,steps,total_reward,timing)
    if logger !== nothing && logger.terminal !== nothing
        println(logger.terminal,"Stopped: ",reason,"; steps=",steps,"; total reward=",total_reward)
        println(logger.terminal,"N-step computation time (excludes reported compilation, logging and rendering): ",
            timing.computation_seconds," s; residual compilation: ",timing.compilation_seconds," s")
        flush(logger.terminal)
    end
    return (; history,final_state=s,final_belief=b,reason,steps,total_reward,timing,
        terminated=reason in (:goal,:collision),truncated=reason == :horizon)
end

"""
    simulate_robust(pomdp, policy; H=remaining_episode_budget, N_f=2,
        seed=42, initial_belief=ExactAdaptiveBelief(pomdp), initial_state=...,
        fault_steps=nothing, fault_actions=nothing, io=stdout,
        log_path=nothing, reward_csv_path=nothing, reward_chart_path=nothing,
        detailed_log=true, top_k=3)

Connect the robust policy, its exact updater, and hidden physical faults.
Uses independent schedule, fault-choice, and observation RNG streams. Planning's
HDR RNG is owned by the value function and never used by the physical simulator.
Finite RobustPolicy retains H=remaining planning horizon and expects a fitted
value function. RecedingHorizonRobustPolicy has its own episode H and maximum
lookahead L; it fits each local problem online, preserving the exact belief.
Use a fresh/reset policy per episode. Both modes enforce their remaining episode
budget; local lookahead never changes physical fault scheduling or termination.
JSONL logs are flushed throughout execution, including failures. Output files
must be distinct and new. SVG plots immediate rewards, never cumulative return
or planner value. All returned true states/schedules are evaluation data only.
"""
function simulate_robust(pomdp::DroneSurveillancePOMDP, policy::AdaptiveRobustPolicy;
        H::Integer=remaining_episode_steps(policy), N_f::Integer=2,
        seed::Integer=42, initial_belief::ExactAdaptiveBelief=ExactAdaptiveBelief(pomdp),
        initial_state::AdaptiveState=initial_adaptive_state(pomdp),
        fault_steps=nothing, fault_actions=nothing, io::Union{Nothing,IO}=stdout,
        log_path=nothing, reward_csv_path=nothing, reward_chart_path=nothing,
        detailed_log::Bool=true, top_k::Integer=3, timing_warmup_completed::Bool=false)
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    top_k > 0 || throw(ArgumentError("top_k must be positive"))
    vf = policy.value_function
    validate_episode_policy(pomdp,policy,H)
    !has_pending_model(policy) ||
        throw(ArgumentError("policy has a pending decision; finish it or reset explicitly"))
    planning_support(vf,initial_belief)
    haskey(initial_belief._logprobs,freeze_hyperstate(initial_state)) ||
        throw(ArgumentError("initial physical hyperstate must be in the initial belief support"))
    termination_reason(pomdp,initial_state,0,H) # preflight state/H validation
    paths = [abspath(p) for p in (log_path,reward_csv_path,reward_chart_path) if p !== nothing]
    length(unique(paths)) == length(paths) || throw(ArgumentError("output paths must be distinct"))
    any(ispath,paths) && throw(ArgumentError("output files already exist; choose a new output directory"))
    reward_chart_path === nothing || endswith(lowercase(reward_chart_path),".svg") ||
        throw(ArgumentError("reward chart must have an .svg extension"))
    master = MersenneTwister(seed)
    schedule_rng, fault_rng, observation_rng = (MersenneTwister(rand(master,UInt)) for _ in 1:3)
    env = HiddenFaultEnvironment(H,N_f; fault_steps,fault_actions,initial_mode=initial_state.mode,
        schedule_rng,fault_rng)
    for path in paths
        mkpath(dirname(path))
    end
    log_file = log_path === nothing ? nothing : open(log_path,"w")
    reward_file = nothing
    try
        reward_file = reward_csv_path === nothing ? nothing : open(reward_csv_path,"w")
        logger = io === nothing && log_file === nothing ? nothing :
            EpisodeLogger(io,log_file; details=detailed_log,top_k=Int(top_k))
        episode_event(logger,:configuration; seed,N_f,episode_horizon=H,
            timing_warmup_completed,
            planning_horizon=vf.horizon,planning_lookahead=vf.horizon,
            effective_lookahead=planning_window(policy),
            planning_mode=planning_mode(policy),
            tied_action_rdn=policy.tied_action_rdn,tie_seed=policy.tie_seed,
            local_terminal_value=policy isa RecedingHorizonRobustPolicy ? :zero : :episode_terminal_zero,
            epsilon=vf.epsilon,hdr_samples=vf.n_samples,hdr_seed=vf.seed,
            value_approximation=:finite_continuations,filtering=filtering_mode(policy),
            particle_count=particle_budget(policy),
            map=(;size=pomdp.size,start=Tuple(pomdp.region_A),goal=Tuple(pomdp.region_B),
                  obstacles=Tuple.(pomdp.obstacles),camera=[collect(r) for r in eachrow(pomdp.camera.confusion)]))
        result = simulate_adaptive(pomdp,policy,updater(policy),initial_belief;
            H,environment=env,initial_state,rng=observation_rng,logger,reward_io=reward_file)
        reward_chart_path === nothing || save_reward_chart(reward_chart_path,result.history)
        return (; result...,fault_steps=env.fault_steps,log_path=log_path === nothing ? nothing : abspath(log_path),
            reward_csv_path=reward_csv_path === nothing ? nothing : abspath(reward_csv_path),
            reward_chart_path=reward_chart_path === nothing ? nothing : abspath(reward_chart_path))
    finally
        log_file === nothing || close(log_file)
        reward_file === nothing || close(reward_file)
    end
end
