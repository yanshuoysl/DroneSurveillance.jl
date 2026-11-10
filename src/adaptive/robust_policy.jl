"""
Finite-horizon policy using the paper-layout robust decision. Its matching
`updater(policy)` conditions on the selected model and advances the external
decision stage only after a successful observation update. Planning receives
only a belief; no physical simulator state, fault schedule, or true mode.

The value function should be fitted at planning points first. Action selection
does not silently fit, shorten the horizon, or replace missing continuations
with a posterior-mean model. Unfitted layers retain their explicit rotate-policy
lower bound. Repeated action queries for the same pending prior reuse its model.
Optional tied_action_rdn=true samples uniformly among numerical maxima using
a separate RNG seeded by tie_seed (default vf.seed). Reset restarts that stream.
"""
mutable struct RobustPolicy <: POMDPs.Policy
    value_function::RobustValueFunction
    stage::Int
    _initial_stage::Int
    _exact_updater::ExactAdaptiveUpdater
    _pending::Union{Nothing,RobustDecision}
    tied_action_rdn::Bool
    tie_seed::Int
    _tie_rng::MersenneTwister
end

function RobustPolicy(vf::RobustValueFunction; stage::Integer=0,
        tied_action_rdn::Bool=false,tie_seed::Integer=vf.seed)
    check_planning_stage(vf,stage)
    tie_seed >= 0 || throw(ArgumentError("tie seed must be nonnegative"))
    return RobustPolicy(vf,Int(stage),Int(stage),ExactAdaptiveUpdater(vf.pomdp),nothing,
        tied_action_rdn,Int(tie_seed),MersenneTwister(tie_seed))
end

struct RobustPolicyUpdater <: POMDPs.Updater
    policy::RobustPolicy
end

POMDPs.updater(policy::RobustPolicy) = RobustPolicyUpdater(policy)
Base.show(io::IO, p::RobustPolicy) = print(io,"RobustPolicy(stage=",p.stage,
    ", horizon=",p.value_function.horizon,", awaiting_observation=",p._pending !== nothing,")")
Base.show(io::IO, ::MIME"text/plain", p::RobustPolicy) = show(io,p)

function POMDPs.action(policy::RobustPolicy, b::ExactAdaptiveBelief)
    vf = policy.value_function
    check_planning_stage(vf,policy.stage)
    if policy._pending !== nothing
        decision = policy._pending
        decision.stage == policy.stage || throw(ArgumentError("stage changed with a pending decision; reset the policy explicitly"))
        validate_belief_selection(vf.pomdp,b,decision.action,decision.selected_rows)
        return decision.action
    end
    candidate_rng = copy(policy._tie_rng)
    decision = robust_decision(vf.pomdp,b,policy.stage,vf;
        tied_action_rdn=policy.tied_action_rdn,rng=candidate_rng)
    decision.action === nothing && throw(ArgumentError("no action at a terminal belief or planning horizon"))
    prepare_decision!(policy._exact_updater,b,decision)
    policy._tie_rng = candidate_rng
    policy._pending = decision
    return decision.action
end

function POMDPs.update(up::RobustPolicyUpdater, b::ExactAdaptiveBelief,
                       action::Integer, observation::DSObservation)
    policy = up.policy
    check_planning_stage(policy.value_function,policy.stage)
    decision = policy._pending
    decision === nothing && throw(ArgumentError("select an action before updating the robust policy belief"))
    decision.stage == policy.stage || throw(ArgumentError("stage changed with a pending decision; reset the policy explicitly"))
    action == decision.action || throw(ArgumentError("observation update action differs from the planned action"))
    posterior = update(policy._exact_updater,b,action,observation)
    policy.stage += 1
    policy._pending = nothing
    return posterior
end

"""Reset an episode's external stage and clear its pending selected model."""
function reset_robust_policy!(policy::RobustPolicy; stage::Integer=policy._initial_stage)
    check_planning_stage(policy.value_function,stage)
    policy.stage = policy._initial_stage = Int(stage)
    policy._pending = nothing
    policy._exact_updater._pending = nothing
    policy._tie_rng = MersenneTwister(policy.tie_seed)
    return policy
end

function POMDPs.initialize_belief(up::RobustPolicyUpdater, initial)
    check_planning_stage(up.policy.value_function,up.policy._initial_stage)
    b = initialize_belief(up.policy._exact_updater,initial)
    reset_robust_policy!(up.policy)
    return b
end

"""
    RecedingHorizonRobustPolicy(template; H, stage=0, tied_action_rdn=false, tie_seed=template.seed)

Explicit receding-horizon approximation: H caps the episode, while
`template.horizon` is the maximum lookahead L. Before each new action, fit a fresh
local value function from the CURRENT exact belief, with horizon min(L,H-stage).
The template supplies the model, HDR calibration, and resource settings; existing
fitted collections are not transferred across local problems. The current
zero-valued terminal layer is retained, so a short window can miss distant goals.

Only one action is executed per local plan. Repeated action queries reuse the
pending model. The matching updater uses those same pre-observation rows and
advances the episode stage only after successful filtering. Counts live in the
belief and are never reset when replanning. No simulator truth is an input.
Optional tied_action_rdn=true samples uniformly among numerical maxima. Its
separate seeded RNG persists across local plans; reset restarts the stream.

With optional `particles=N`, filter exactly over the retained weighted support
then compress mode/count histories within configuration strata. Configuration
marginals and configuration support are preserved for that update. N must cover all
map configurations. The ExactAdaptiveBelief container then represents a weighted
particle approximation, not an exact posterior; the default remains exact.
"""
mutable struct RecedingHorizonRobustPolicy <: POMDPs.Policy
    value_function::RobustValueFunction
    episode_horizon::Int
    stage::Int
    _initial_stage::Int
    _policy::Union{Nothing,RobustPolicy}
    _pending_stage::Union{Nothing,Int}
    _fit_report::Union{Nothing,NamedTuple}
    particle_count::Union{Nothing,Int}
    particle_seed::Int
    _particle_rng::MersenneTwister
    _filter_report::Union{Nothing,NamedTuple}
    tied_action_rdn::Bool
    tie_seed::Int
    _tie_rng::MersenneTwister
end

function RecedingHorizonRobustPolicy(template::RobustValueFunction;
        H::Integer, stage::Integer=0, particles::Union{Nothing,Integer}=nothing,
        particle_seed::Integer=template.seed,
        tied_action_rdn::Bool=false,tie_seed::Integer=template.seed)
    check_planning_stage(template,0)
    template.horizon > 0 || throw(ArgumentError("planning lookahead L must be positive"))
    0 <= H <= typemax(Int) || throw(ArgumentError("episode H must be nonnegative and fit in Int"))
    0 <= stage <= H || throw(ArgumentError("episode stage must lie in 0:H"))
    particle_seed >= 0 || throw(ArgumentError("particle seed must be nonnegative"))
    tie_seed >= 0 || throw(ArgumentError("tie seed must be nonnegative"))
    if particles !== nothing
        # Reserve one representative per possible configuration to preserve
        # pose marginals, independently of hidden modes/counts.
        minimum_particles = (prod(template.pomdp.size)-length(template.pomdp.obstacles))*length(HEADINGS)+1
        particles >= minimum_particles || throw(ArgumentError("particle budget must be at least $minimum_particles to preserve configuration support"))
    end
    return RecedingHorizonRobustPolicy(template,Int(H),Int(stage),Int(stage),nothing,nothing,nothing,
        particles === nothing ? nothing : Int(particles),Int(particle_seed),MersenneTwister(particle_seed),nothing,
        tied_action_rdn,Int(tie_seed),MersenneTwister(tie_seed))
end

struct RecedingHorizonRobustUpdater <: POMDPs.Updater
    policy::RecedingHorizonRobustPolicy
end
POMDPs.updater(policy::RecedingHorizonRobustPolicy) = RecedingHorizonRobustUpdater(policy)

function check_receding_stage(policy::RecedingHorizonRobustPolicy)
    check_planning_stage(policy.value_function,0)
    policy.value_function.horizon > 0 || throw(ArgumentError("planning lookahead must be positive"))
    0 <= policy.stage <= policy.episode_horizon || throw(ArgumentError("episode stage must lie in 0:H"))
    if policy._pending_stage !== nothing
        policy._pending_stage == policy.stage ||
            throw(ArgumentError("episode stage changed with a pending model; reset explicitly"))
        policy._policy !== nothing && policy._policy._pending !== nothing ||
            throw(ArgumentError("missing pending local policy"))
        policy._policy.value_function.horizon == min(policy.value_function.horizon,
            policy.episode_horizon-policy.stage) ||
            throw(ArgumentError("episode horizon changed with a pending model; reset explicitly"))
    end
    return nothing
end

function POMDPs.action(policy::RecedingHorizonRobustPolicy, b::ExactAdaptiveBelief)
    check_receding_stage(policy)
    policy.particle_count === nothing || length(b) <= policy.particle_count ||
        throw(ArgumentError("initial belief exceeds particle budget; initialize it with the matching updater"))
    if policy._pending_stage !== nothing
        return action(policy._policy,b) # validates the same prior, reuses theta
    end
    policy.stage < policy.episode_horizon || throw(ArgumentError("episode horizon reached"))
    all(s -> s.outcome != ACTIVE_STATE,keys(b._logprobs)) &&
        throw(ArgumentError("no action at a terminal belief"))
    template = policy.value_function
    local_horizon = min(template.horizon,policy.episode_horizon-policy.stage)
    local_vf = RobustValueFunction(template.pomdp;horizon=local_horizon,
        epsilon=template.epsilon,n_samples=template.n_samples,seed=template.seed,
        options=template.options)
    report = fit_value_function!(local_vf,b)
    local_policy = RobustPolicy(local_vf;tied_action_rdn=policy.tied_action_rdn,tie_seed=policy.tie_seed)
    local_policy._tie_rng = copy(policy._tie_rng)
    a = action(local_policy,b)
    # Commit only after fitting and row selection succeed.
    policy._policy = local_policy
    policy._pending_stage = policy.stage
    policy._tie_rng = copy(local_policy._tie_rng)
    policy._fit_report = (; points=report.points,updates=report.updates,
        sweeps=report.sweeps,approximation=report.approximation)
    return a
end

function POMDPs.update(up::RecedingHorizonRobustUpdater,b::ExactAdaptiveBelief,
                       a::Integer,o::DSObservation)
    policy = up.policy
    check_receding_stage(policy)
    policy._pending_stage === nothing && throw(ArgumentError("select an action before updating"))
    if policy.particle_count === nothing
        posterior = update(updater(policy._policy),b,a,o)
        policy._filter_report = nothing
    else
        local_policy = policy._policy
        pending = local_policy._pending
        a == pending.action || throw(ArgumentError("observation action differs from the planned action"))
        pending.stage == local_policy.stage || throw(ArgumentError("local stage changed with a pending model"))
        local_policy._exact_updater._pending === pending.selected_rows ||
            throw(ArgumentError("pending updater model differs from the planned model"))
        expanded = exact_belief_update(policy.value_function.pomdp,b,a,o,pending.selected_rows)
        candidate_rng = deepcopy(policy._particle_rng)
        sampled = configuration_stratified_particles(expanded,policy.particle_count;rng=candidate_rng)
        posterior = sampled.belief
        # Commit sampling and both updater clocks only after success.
        policy._particle_rng = candidate_rng
        local_policy._exact_updater._pending = nothing
        local_policy._pending = nothing
        local_policy.stage += 1
        policy._filter_report = (;method=:configuration_stratified_particles,
            budget=policy.particle_count,expanded=sampled.expanded,retained=sampled.retained,
            configurations=sampled.configurations,resampled=sampled.resampled)
    end
    policy.stage += 1
    policy._pending_stage = nothing
    return posterior
end

function reset_robust_policy!(policy::RecedingHorizonRobustPolicy;
                              stage::Integer=policy._initial_stage)
    check_planning_stage(policy.value_function,0)
    0 <= stage <= policy.episode_horizon || throw(ArgumentError("episode stage must lie in 0:H"))
    policy.stage = policy._initial_stage = Int(stage)
    policy._policy = nothing
    policy._pending_stage = nothing
    policy._fit_report = nothing
    policy._particle_rng = MersenneTwister(policy.particle_seed)
    policy._filter_report = nothing
    policy._tie_rng = MersenneTwister(policy.tie_seed)
    return policy
end

function POMDPs.initialize_belief(up::RecedingHorizonRobustUpdater,initial)
    b = initialize_belief(ExactAdaptiveUpdater(up.policy.value_function.pomdp),initial)
    sampled = nothing
    candidate_rng = MersenneTwister(up.policy.particle_seed)
    if up.policy.particle_count !== nothing
        sampled = configuration_stratified_particles(b,up.policy.particle_count;rng=candidate_rng)
        b = sampled.belief
    end
    reset_robust_policy!(up.policy)
    if sampled !== nothing
        up.policy._particle_rng = candidate_rng
        up.policy._filter_report = (;method=:configuration_stratified_particles,
            budget=up.policy.particle_count,expanded=sampled.expanded,retained=sampled.retained,
            configurations=sampled.configurations,resampled=sampled.resampled)
    end
    return b
end

Base.show(io::IO,p::RecedingHorizonRobustPolicy) = print(io,
    "RecedingHorizonRobustPolicy(stage=",p.stage,", H=",p.episode_horizon,
    ", L=",p.value_function.horizon,", awaiting_observation=",p._pending_stage !== nothing,")")
Base.show(io::IO,::MIME"text/plain",p::RecedingHorizonRobustPolicy) = show(io,p)

# Shared episode adapters; finite-horizon behavior remains available explicitly.
const AdaptiveRobustPolicy = Union{RobustPolicy,RecedingHorizonRobustPolicy}
remaining_episode_steps(p::RobustPolicy) = p.value_function.horizon-p.stage
remaining_episode_steps(p::RecedingHorizonRobustPolicy) = p.episode_horizon-p.stage
planning_mode(::RobustPolicy) = :finite_horizon
planning_mode(::RecedingHorizonRobustPolicy) = :receding_horizon
planning_window(p::RobustPolicy) = p.value_function.horizon-p.stage
planning_window(p::RecedingHorizonRobustPolicy) = min(p.value_function.horizon,remaining_episode_steps(p))
pending_robust_decision(p::RobustPolicy) = p._pending
pending_robust_decision(p::RecedingHorizonRobustPolicy) =
    p._policy === nothing ? nothing : pending_robust_decision(p._policy)
has_pending_model(p::RobustPolicy) = p._pending !== nothing || p._exact_updater._pending !== nothing
has_pending_model(p::RecedingHorizonRobustPolicy) = p._pending_stage !== nothing ||
    (p._policy !== nothing && has_pending_model(p._policy))
policy_fit_summary(::RobustPolicy) = nothing
policy_fit_summary(p::RecedingHorizonRobustPolicy) = p._fit_report

function validate_episode_policy(pomdp,policy::AdaptiveRobustPolicy,H)
    policy.value_function.pomdp === pomdp || throw(ArgumentError("policy belongs to another model"))
    if policy isa RobustPolicy
        check_planning_stage(policy.value_function,policy.stage)
    else
        check_receding_stage(policy)
    end
    H == remaining_episode_steps(policy) || throw(ArgumentError("H must equal the remaining EPISODE budget"))
    return nothing
end

filtering_mode(::RobustPolicy) = :exact_selected_model
filtering_mode(p::RecedingHorizonRobustPolicy) = p.particle_count === nothing ?
    :exact_selected_model : :configuration_stratified_particles
particle_budget(::RobustPolicy) = nothing
particle_budget(p::RecedingHorizonRobustPolicy) = p.particle_count
particle_filter_report(::RobustPolicy) = nothing
particle_filter_report(p::RecedingHorizonRobustPolicy) = p._filter_report
