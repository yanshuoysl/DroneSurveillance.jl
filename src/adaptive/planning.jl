# Finite-horizon point-based value layers. The exact belief remains unchanged.

struct RobustPlanningError <: Exception
    kind::Symbol
    message::String
end
Base.showerror(io::IO, e::RobustPlanningError) = print(io, e.kind, ": ", e.message)

"""Numerical tolerances and explicit resource limits; exceeding a limit throws."""
Base.@kwdef struct RobustPlannerOptions
    gap_tolerance::Float64 = 1e-5
    max_iterations::Int = 10_000
    dual_step_scale::Float64 = 1.0
    max_hypotheses::Int = 5000
    max_supports::Int = 1000
    max_cached_actions::Int = 6000
    max_cached_rows::Int = 50_000
    max_edges::Int = 50_000
    max_total_edges::Int = 200_000
    max_score_entries::Int = 2_000_000
    max_saddle_columns::Int = 128
    max_vectors::Int = 128
    max_training_points::Int = 5000
end

"""An immutable continuation coefficient vector on one full-hyperstate support."""
struct RobustContinuation
    stage::Int
    support::Tuple
    coefficients::Tuple{Vararg{Float64}}
    action::Union{Nothing,Int}
end

struct PreparedRobustAction
    support::Tuple
    action::Int
    parents::Vector{AdaptiveState}
    ranges::Vector{UnitRange{Int}}
    hdrs::Vector{Union{Nothing,DirichletHDR}}
    children::Vector{BeliefHyperstate}
    rewards::Vector{Float64}
    log_likelihoods::Matrix{Float64}
    observations::Vector{DSObservation}
    next_supports::Vector{Tuple}
end

"""
Finite continuation approximation of the paper's V_n(b), n=0,...,horizon.
`value(vf,b,n)` evaluates max_alpha sum_s b(s)*alpha(s), with V_N=0.
The baseline is an always-rotate policy (reward -1 until the horizon); last-step
collections include all valid actions. Earlier collections improve at supplied
planning beliefs through `fit_value_function!`. No belief particles are used.

The model, calibration settings, and internal caches are read-only after creation.
`seed` restarts calibration for each count row, making thresholds reproducible
and independent of cache traversal order. A backup never resamples a threshold.
"""
mutable struct RobustValueFunction
    pomdp::DroneSurveillancePOMDP
    horizon::Int
    epsilon::Float64
    n_samples::Int
    seed::Int
    options::RobustPlannerOptions
    _signature::Tuple
    _families::Dict{Tuple{Int,Vector{BeliefHyperstate}},Vector{RobustContinuation}}
    _last_complete::Set{Vector{BeliefHyperstate}}
    _hdrs::Dict{Tuple{BeliefHyperstate,Int},DirichletHDR}
    _actions::Dict{Tuple{Tuple,Int,Bool},PreparedRobustAction}
    _immediate::Dict{Tuple{BeliefHyperstate,Int},HDRRowMinimum}
end

Base.show(io::IO, vf::RobustValueFunction) = print(io, "RobustValueFunction(horizon=",vf.horizon,
    ", supports=",length(vf._families),", cached_rows=",length(vf._hdrs),", approximation=:finite_continuations)")
Base.show(io::IO, ::MIME"text/plain", vf::RobustValueFunction) = show(io,vf)
Base.show(io::IO, a::RobustContinuation) = print(io,"RobustContinuation(stage=",a.stage,
    ", hypotheses=",length(a.support),", action=",a.action,")")

planning_model_signature(p) = (p.size, p.region_A, p.region_B, p.initial_heading,
    Tuple(sort!(collect(Tuple.(p.obstacles)))), Tuple(p.camera.confusion), discount(p))

function RobustValueFunction(pomdp::DroneSurveillancePOMDP; horizon::Integer,
                              epsilon::Real, n_samples::Integer=K, seed::Integer=42,
                              options::RobustPlannerOptions=RobustPlannerOptions())
    horizon >= 0 || throw(ArgumentError("planning horizon must be nonnegative"))
    seed >= 0 || throw(ArgumentError("calibration seed must be nonnegative"))
    check_hdr_settings(epsilon, n_samples)
    isfinite(discount(pomdp)) && 0 <= discount(pomdp) <= 1 ||
        throw(ArgumentError("planning discount must lie in [0,1]"))
    isfinite(options.gap_tolerance) && options.gap_tolerance > 0 ||
        throw(ArgumentError("gap_tolerance must be finite and positive"))
    isfinite(options.dual_step_scale) && 0 < options.dual_step_scale < 2 ||
        throw(ArgumentError("dual_step_scale must lie in (0,2)"))
    for field in (:max_iterations, :max_hypotheses, :max_supports, :max_cached_actions, :max_cached_rows,
                  :max_edges, :max_total_edges, :max_score_entries, :max_saddle_columns,
                  :max_vectors, :max_training_points)
        getfield(options, field) > 0 || throw(ArgumentError("$field must be positive"))
    end
    return RobustValueFunction(pomdp, Int(horizon), Float64(epsilon), Int(n_samples),
        Int(seed), options, (planning_model_signature(pomdp),Int(horizon),Float64(epsilon),Int(n_samples),Int(seed),options),
        Dict{Tuple{Int,Vector{BeliefHyperstate}},Vector{RobustContinuation}}(), Set{Vector{BeliefHyperstate}}(),
        Dict{Tuple{BeliefHyperstate,Int},DirichletHDR}(),
        Dict{Tuple{Tuple,Int,Bool},PreparedRobustAction}(),
        Dict{Tuple{BeliefHyperstate,Int},HDRRowMinimum}())
end

function check_planning_stage(vf, n)
    (planning_model_signature(vf.pomdp),vf.horizon,vf.epsilon,vf.n_samples,vf.seed,vf.options) == vf._signature ||
        throw(RobustPlanningError(:stale_model, "model/calibration settings changed; construct a new value function and HDR caches"))
    n isa Integer && 0 <= n <= vf.horizon || throw(ArgumentError("stage must lie in 0:horizon"))
    return nothing
end

planning_key_order(s::BeliefHyperstate) = (Tuple(s.config.quad)..., s.config.heading,
    mode_index(s.mode), Int(s.outcome), Tuple((count_key_order(first(p)), last(p)) for p in s.increments))
planning_support(keys) = Tuple(sort!(unique(collect(keys)); by=planning_key_order))

function planning_support(vf, b::ExactAdaptiveBelief)
    length(b) <= vf.options.max_hypotheses ||
        throw(RobustPlanningError(:resource, "belief exceeds max_hypotheses; no hypotheses were pruned"))
    for s in support(b)
        validate_belief_state(vf.pomdp, s)
    end
    return planning_support(keys(b._logprobs))
end

function planning_weights(b, states)
    weights = [exp(b._logprobs[s]) for s in states]
    return weights ./ sum(weights)
end

function support_belief(vf, states)
    masses = Dict(s => 0.0 for s in states)
    return ExactAdaptiveBelief(masses, Val(:logmass))
end

function planning_hdr!(vf, parent, action)
    !haskey(vf._hdrs,(parent,action)) && length(vf._hdrs) >= vf.options.max_cached_rows &&
        throw(RobustPlanningError(:resource,"max_cached_rows reached"))
    return get!(vf._hdrs, (parent, action)) do
        build_row_hdr(vf.pomdp, thaw_hyperstate(parent), action; epsilon=vf.epsilon,
                      n_samples=vf.n_samples, rng=MersenneTwister(vf.seed))
    end
end

function prepare_robust_action!(vf, states, action; need_observations::Bool=true)
    cache_key = (states, Int(action),need_observations)
    haskey(vf._actions, cache_key) && return vf._actions[cache_key]
    length(vf._actions) < vf.options.max_cached_actions ||
        throw(RobustPlanningError(:resource, "max_cached_actions reached"))
    action in actions(vf.pomdp, support_belief(vf, states)) ||
        throw(ArgumentError("action is unavailable for this belief support"))
    parents = AdaptiveState[thaw_hyperstate(s) for s in states]
    ranges = UnitRange{Int}[]
    hdrs = Union{Nothing,DirichletHDR}[]
    children = BeliefHyperstate[]
    rewards = Float64[]
    for (key, s) in zip(states, parents)
        first_index = length(children) + 1
        # Support/count expansion does not require HDR calibration. Check its
        # resource cost before constructing potentially thousands of HDRs.
        successors = feasible_successors(vf.pomdp,s.config,s.mode,action)
        length(children) + length(successors) <= vf.options.max_edges ||
            throw(RobustPlanningError(:resource, "action expansion exceeds max_edges"))
        for (rp, mp) in successors
            counts = parameter_update_transition(vf.pomdp, s, action, rp, mp).val
            child = adaptive_state(vf.pomdp, rp, mp, counts)
            push!(children, freeze_hyperstate(child))
            push!(rewards, adaptive_reward(vf.pomdp, s, action, child))
        end
        push!(ranges, first_index:length(children))
    end
    sum(length(p.children) for p in values(vf._actions); init=0) + length(children) <= vf.options.max_total_edges ||
        throw(RobustPlanningError(:resource, "action caches exceed max_total_edges"))
    if need_observations
        # Structural support uses log likelihoods, including probabilities too small
        # for exp(logp) to be represented. Masks never discard such hypotheses.
        logs = [belief_camera_loglikelihood(vf.pomdp, s.config, s.outcome, o)
                for s in children, o in ALL_OBSERVATIONS]
        indices = [z for z in axes(logs, 2) if any(isfinite, view(logs, :, z))]
        observations = DSObservation[ALL_OBSERVATIONS[z] for z in indices]
        logs = logs[:, indices]
        # Many noisy camera observations have exactly the same structural support.
        # Sort/merge that set once, and fail on an oversized set immediately rather
        # than constructing the same large tuple for all 64 camera triples.
        support_masks = Dict{BitSet,Tuple}()
        next_supports = Tuple[]
        for z in axes(logs,2)
            mask = BitSet(e for e in eachindex(children) if isfinite(logs[e,z]))
            next = get!(support_masks,mask) do
                # Enforce the cap before sorting or constructing a potentially huge
                # tuple type, which itself can consume substantial compile time.
                unique_children = unique(BeliefHyperstate[children[e] for e in mask])
                length(unique_children) <= vf.options.max_hypotheses ||
                    throw(RobustPlanningError(:resource,
                        "successor support has $(length(unique_children)) hypotheses; max_hypotheses=$(vf.options.max_hypotheses); no pruning"))
                Tuple(sort!(unique_children;by=planning_key_order))
            end
            push!(next_supports,next)
        end
    else
        # V_N=0: the final decision needs only immediate rewards and HDR rows.
        observations = DSObservation[]
        logs = zeros(length(children),0)
        next_supports = Tuple[]
    end
    for (key,s) in zip(states,parents)
        push!(hdrs,is_terminal(s) ? nothing : planning_hdr!(vf,key,action))
    end
    prepared = PreparedRobustAction(states, Int(action), parents, ranges, hdrs,
                                   children, rewards, logs, observations, next_supports)
    vf._actions[cache_key] = prepared
    return prepared
end

function planner_row_minimum(vf, prepared, i, coefficients)
    hdr = prepared.hdrs[i]
    hdr === nothing && return nothing
    key = (prepared.support[i], prepared.action)
    immediate = coefficients == prepared.rewards[prepared.ranges[i]]
    immediate && haskey(vf._immediate, key) && return vf._immediate[key]
    result = minimize_hdr_row(hdr, coefficients; log_atol=1e-8,
        objective_atol=min(1e-10, vf.options.gap_tolerance * 1e-3), objective_rtol=1e-12)
    immediate && (vf._immediate[key] = result)
    return result
end

function planner_immediate_minimum!(vf, parent::BeliefHyperstate, action)
    parent.outcome == ACTIVE_STATE || return nothing
    key = (parent, Int(action))
    return get!(vf._immediate, key) do
        hdr = planning_hdr!(vf, parent, action)
        source = thaw_hyperstate(parent)
        # At the final decision V_N=0. Rewards do not depend on destination
        # counts, so neither count-history children nor an action/observation
        # cache is needed. Reuse each row minimum across overlapping supports.
        coefficients = [adaptive_reward(vf.pomdp, source, action,
                            adaptive_state(vf.pomdp, rp, mp))
                        for (rp, mp) in hdr.successors]
        minimize_hdr_row(hdr, coefficients; log_atol=1e-8,
            objective_atol=min(1e-10, vf.options.gap_tolerance * 1e-3), objective_rtol=1e-12)
    end
end

function add_planning_continuation!(vf, continuation::RobustContinuation)
    key = (continuation.stage, collect(BeliefHyperstate, continuation.support))
    family = vf._families[key]
    # Componentwise dominance is safe on this same support and stage.
    any(a -> all(a.coefficients[i] >= continuation.coefficients[i]
                 for i in eachindex(continuation.coefficients)), family) && return false
    keep = [a for a in family if !all(continuation.coefficients[i] >= a.coefficients[i]
                                    for i in eachindex(continuation.coefficients))]
    length(keep) < vf.options.max_vectors ||
        throw(RobustPlanningError(:resource, "max_vectors reached; no continuation was silently dropped"))
    push!(keep, continuation)
    vf._families[key] = keep
    return true
end

function continuation_family!(vf, states, n)
    @nospecialize states
    # Private vector snapshots are read-only cache keys. Their fixed type
    # avoids compiling dictionary lookups and iteration for every tuple length.
    state_vector = collect(BeliefHyperstate, states)
    key = (Int(n), state_vector)
    if !haskey(vf._families, key)
        length(vf._families) < vf.options.max_supports ||
            throw(RobustPlanningError(:resource, "max_supports reached"))
        # A fixed rotate-clockwise policy is always feasible for active cells,
        # never collides/reaches a different cell, and costs -1 per action.
        remaining = vf.horizon - n
        gamma = discount(vf.pomdp)
        baseline = remaining == 0 ? 0.0 : gamma == 1 ? -Float64(remaining) :
                   expm1(remaining * log(gamma)) / (1-gamma)
        coefficients = Tuple(s.outcome == ACTIVE_STATE ? baseline : 0.0 for s in state_vector)
        active = any(s -> s.outcome == ACTIVE_STATE, state_vector)
        vf._families[key] = [RobustContinuation(n, states, coefficients, remaining == 0 || !active ? nothing : 5)]
    end
    if n == vf.horizon-1 && !(state_vector in vf._last_complete)
        for a in actions(vf.pomdp, support_belief(vf, state_vector))
            coefficients = Float64[]
            for parent in state_vector
                result = planner_immediate_minimum!(vf, parent, a)
                push!(coefficients, result === nothing ? 0.0 : result.objective_lower_bound)
            end
            add_planning_continuation!(vf, RobustContinuation(n, states, Tuple(coefficients), a))
        end
        push!(vf._last_complete, state_vector)
    end
    return vf._families[key]
end

"""Evaluate the current finite continuation approximation; no true state input."""
function POMDPs.value(vf::RobustValueFunction, b::ExactAdaptiveBelief, n::Integer)
    check_planning_stage(vf, n)
    states = planning_support(vf, b)
    family = continuation_family!(vf, states, n)
    weights = planning_weights(b, states)
    return maximum(sum(weights .* a.coefficients) for a in family)
end

"""
Improve finite value layers backward at explicit planning beliefs, or expand from
an initial belief under explicitly chosen HDR-mode rows (`expand=true`). Mode
rows generate training points only; action backups still minimize over HDRs.
All structurally possible observations are enumerated with the exact updater.
Resource limits raise errors rather than prune beliefs, horizons, or vectors.
"""
function fit_value_function!(vf::RobustValueFunction, initial::ExactAdaptiveBelief;
                             stage::Integer=0, expand::Bool=true, sweeps::Integer=1)
    check_planning_stage(vf, stage)
    sweeps > 0 || throw(ArgumentError("sweeps must be positive"))
    points = Dict{Int,Vector{ExactAdaptiveBelief}}(Int(stage) => [initial])
    planning_support(vf,initial)
    # Belief dictionaries are immutable snapshots by contract. Compare their
    # contents directly: a tuple of every (state, weight) creates a new Julia
    # key type for each support size and can spend minutes compiling insertion
    # of a large hypothetical belief into this set.
    seen = Set{Tuple{Int,Dict{BeliefHyperstate,Float64}}}()
    push!(seen, (Int(stage), initial._logprobs))
    if expand
        for n in stage:vf.horizon-2
            for b in get(points, n, ExactAdaptiveBelief[]), a in actions(vf.pomdp, b)
                states = planning_support(vf,b)
                prepared = prepare_robust_action!(vf, states, a)
                rows = Dict{AdaptiveState,SelectedTransitionRow}()
                for i in eachindex(states)
                    hdr = prepared.hdrs[i]
                    hdr === nothing && continue
                    d = hdr.distribution.alpha .- 1
                    theta = d ./ sum(d)
                    rows[prepared.parents[i]] = SelectedTransitionRow(theta, hdr)
                end
                selected = BeliefTransitionRows(vf.pomdp, b, a, rows)
                for o in prepared.observations
                    next = exact_belief_update(vf.pomdp, b, a, o, selected)
                    signature = (Int(n+1), next._logprobs)
                    signature in seen && continue
                    length(seen) < vf.options.max_training_points ||
                        throw(RobustPlanningError(:resource, "max_training_points reached"))
                    push!(seen, signature)
                    push!(get!(points, Int(n+1), ExactAdaptiveBelief[]), next)
                end
            end
        end
    end
    return fit_value_function!(vf, points; sweeps)
end

function fit_value_function!(vf::RobustValueFunction, points::AbstractDict; sweeps::Integer=1)
    check_planning_stage(vf,0)
    sweeps > 0 || throw(ArgumentError("sweeps must be positive"))
    all(bs -> bs isa AbstractVector && all(b -> b isa ExactAdaptiveBelief,bs), values(points)) ||
        throw(ArgumentError("planning points must map stages to vectors of ExactAdaptiveBeliefs"))
    sum(length(bs) for bs in values(points); init=0) <= vf.options.max_training_points ||
        throw(RobustPlanningError(:resource, "too many training points"))
    for n in keys(points)
        check_planning_stage(vf,n)
    end
    updates = 0
    for _ in 1:sweeps, n in sort!(collect(keys(points)); rev=true), b in points[n]
        states = planning_support(vf,b)
        continuation_family!(vf, states, n)
        n == vf.horizon && continue
        # The final decision layer already contains every valid action's
        # row-wise robust immediate-reward vector. Further point backups cannot
        # improve that collection; avoid repeating them for each observation's
        # different weights on the same support.
        n == vf.horizon-1 && continue
        decision = robust_decision(vf.pomdp, b, n, vf)
        for result in decision.action_results
            updates += add_planning_continuation!(vf, result.continuation)
        end
    end
    return (; value_function=vf, points=sum(length(bs) for bs in values(points); init=0),
             updates, sweeps=Int(sweeps), approximation=:finite_continuations)
end
