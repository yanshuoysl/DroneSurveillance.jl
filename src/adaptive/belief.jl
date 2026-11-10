# Exact finite-support filtering conditional on explicitly selected HDR rows.

# Immutable keys protect stored hypotheses from mutable DirichletCounts dictionaries.
struct BeliefHyperstate
    config::DSState
    mode::FaultMode
    increments::Tuple{Vararg{Pair{CountKey,Int}}}
    outcome::AdaptiveOutcome
end

Base.:(==)(a::BeliefHyperstate, b::BeliefHyperstate) =
    a.config == b.config && a.mode == b.mode && a.increments == b.increments && a.outcome == b.outcome
Base.isequal(a::BeliefHyperstate, b::BeliefHyperstate) =
    isequal(a.config, b.config) && isequal(a.mode, b.mode) &&
    isequal(a.increments, b.increments) && a.outcome == b.outcome
Base.hash(s::BeliefHyperstate, h::UInt) = hash((s.config, s.mode, s.increments, s.outcome), h)

count_key_order(k::CountKey) =
    (Tuple(k.config.quad)..., k.config.heading, mode_index(k.mode), k.action,
     Tuple(k.next_config.quad)..., k.next_config.heading, mode_index(k.next_mode))

function freeze_hyperstate(s::AdaptiveState)
    counts = DirichletCounts(s.counts.increments)
    entries = sort!(collect(counts.increments); by=pair -> count_key_order(first(pair)))
    return BeliefHyperstate(s.config, s.mode, Tuple(entries), s.outcome)
end

function thaw_hyperstate(s::BeliefHyperstate)
    counts = DirichletCounts(Dict{CountKey,Int}(s.increments))
    return AdaptiveState(s.config, s.mode, counts; outcome=s.outcome)
end

function validate_belief_state(pomdp, s::AdaptiveState)
    canonical = adaptive_state(pomdp, s.config, s.mode)
    s.config == canonical.config && s.outcome == canonical.outcome ||
        throw(ArgumentError("belief state configuration or outcome is inconsistent with the map"))
    return nothing
end

belief_logadd(a, b) = a == -Inf ? b : b == -Inf ? a :
    max(a, b) + log1p(exp(-abs(a - b)))

function belief_logtotal(masses)
    isempty(masses) && throw(ArgumentError("a belief needs at least one positive-mass hypothesis"))
    return foldl(belief_logadd, values(masses); init=-Inf)
end

"""
    ExactAdaptiveBelief(pomdp, entries; logweights=false)
    ExactAdaptiveBelief(pomdp, state=initial_adaptive_state(pomdp))

A finite, exact-enumeration belief over full AdaptiveStates. `entries` contains
`state => weight` pairs (or `(state, weight)` tuples); duplicate full states are
merged and weights normalized. Pass logweights=true for logarithmic masses,
including very small probabilities. Zero mass is omitted, with no threshold
pruning. Internal keys contain immutable count snapshots and weights are stored
in log space. Internal fields are read-only implementation details.

`support(b)` and iteration return independent state/count snapshots. Iteration
yields `(state, probability)`; pdf gives probabilities and logpdf gives their
logarithms. Probability views can underflow for extremely small masses, but such
hypotheses remain in the stored support and in subsequent exact updates.
"""
struct ExactAdaptiveBelief
    _logprobs::Dict{BeliefHyperstate,Float64}

    function ExactAdaptiveBelief(masses::Dict{BeliefHyperstate,Float64}, ::Val{:logmass})
        all(isfinite, values(masses)) || throw(ArgumentError("stored log masses must be finite"))
        total = belief_logtotal(masses)
        normalized = Dict(k => v - total for (k, v) in masses)
        all(isfinite, values(normalized)) ||
            throw(ArgumentError("normalized log probabilities exceed Float64 range"))
        return new(normalized)
    end
end

function ExactAdaptiveBelief(pomdp::DroneSurveillancePOMDP, entries; logweights::Bool=false)
    masses = Dict{BeliefHyperstate,Float64}()
    for (s, weight) in entries
        s isa AdaptiveState || throw(ArgumentError("belief entries must contain AdaptiveStates"))
        weight isa Real || throw(ArgumentError("belief weights must be real"))
        validate_belief_state(pomdp, s)
        w = Float64(weight)
        if logweights
            w == -Inf && continue
            isfinite(w) || throw(ArgumentError("log weights must be finite or -Inf"))
            logw = w
        else
            isfinite(w) && w >= 0 || throw(ArgumentError("weights must be finite and nonnegative"))
            w == 0 && continue
            logw = log(w)
        end
        key = freeze_hyperstate(s)
        masses[key] = belief_logadd(get(masses, key, -Inf), logw)
    end
    return ExactAdaptiveBelief(masses, Val(:logmass))
end

ExactAdaptiveBelief(pomdp::DroneSurveillancePOMDP,
                   s::AdaptiveState=initial_adaptive_state(pomdp)) =
    ExactAdaptiveBelief(pomdp, [s => 1.0])

Base.length(b::ExactAdaptiveBelief) = length(b._logprobs)
Distributions.support(b::ExactAdaptiveBelief) = [thaw_hyperstate(s) for s in keys(b._logprobs)]
Distributions.logpdf(b::ExactAdaptiveBelief, s::AdaptiveState) =
    get(b._logprobs, freeze_hyperstate(s), -Inf)
Distributions.pdf(b::ExactAdaptiveBelief, s::AdaptiveState) = exp(Distributions.logpdf(b, s))

function Base.iterate(b::ExactAdaptiveBelief, args...)
    item = iterate(b._logprobs, args...)
    item === nothing && return nothing
    (s, logw), cursor = item
    return ((thaw_hyperstate(s), exp(logw)), cursor)
end

function POMDPs.actions(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief)
    has_active = false
    # Validate every hypothesis, but never intersect pose-dependent action sets.
    for key in keys(b._logprobs)
        s = thaw_hyperstate(key)
        validate_belief_state(pomdp, s)
        is_terminal(s) && continue
        has_active = true
    end
    return has_active ? collect(actions(pomdp)) : Int64[]
end

function require_belief_action(pomdp, b, a::Integer)
    1 <= a <= N_ACTIONS || throw(ArgumentError("action must be from 1 to $N_ACTIONS"))
    # Terminal-only beliefs may be updated through their absorbing kernel.
    if any(k.outcome == ACTIVE_STATE for k in keys(b._logprobs))
        a in actions(pomdp, b) ||
            throw(ArgumentError("action is unavailable for this belief"))
    else
        for s in support(b)
            validate_belief_state(pomdp, s)
        end
    end
    return nothing
end

"""
    SelectedTransitionRow(theta, hdr)

An immutable snapshot of an explicitly chosen row and its HDR. Theta follows
hdr.successors order. Construction does not optimize or select a model. The
parent-specific support and count checks occur when preparing/applying rows.
"""
struct SelectedTransitionRow
    theta::Tuple{Vararg{Float64}}
    successors::Tuple{Vararg{Tuple{DSState,FaultMode}}}
    alpha::Tuple{Vararg{Float64}}
    epsilon::Float64
    n_samples::Int
    tau::Float64

    function SelectedTransitionRow(theta::AbstractVector{<:Real}, hdr::DirichletHDR)
        probabilities = transition_probabilities(theta, length(hdr.successors))
        check_hdr_settings(hdr.epsilon, hdr.n_samples)
        isfinite(hdr.tau) && hdr.tau > 0 || throw(ArgumentError("invalid HDR threshold"))
        contains(hdr, probabilities) || throw(ArgumentError("row is outside its HDR"))
        return new(Tuple(probabilities), Tuple(hdr.successors),
                   Tuple(Float64.(hdr.distribution.alpha)), hdr.epsilon, hdr.n_samples, hdr.tau)
    end
end

function selected_physical_transition(pomdp, s, a, row::SelectedTransitionRow)
    hdr = DirichletHDR(hdr_distribution(collect(row.alpha)), collect(row.successors),
                       row.epsilon, row.n_samples, row.tau)
    return locomotion_mode_transition(pomdp, s, a, collect(row.theta); hdr=hdr)
end

"""
    BeliefTransitionRows(pomdp, belief, action, rows)

Bind an explicitly selected model to a complete prior belief and action. `rows`
maps every active parent AdaptiveState to a SelectedTransitionRow; omit terminal
parents. Missing/extra parents, invalid boundaries, stale counts and unrelated
HDRs are rejected. Immutable snapshots protect selections from caller mutation.
Prepare this before the real observation; this type neither receives an
observation nor chooses a worst-case row. Internal fields are read-only.
"""
struct BeliefTransitionRows
    action::Int64
    _prior::Dict{BeliefHyperstate,Float64}
    _rows::Dict{BeliefHyperstate,SelectedTransitionRow}

    function BeliefTransitionRows(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief,
                                  a::Integer, rows::AbstractDict)
        require_belief_action(pomdp, b, a)
        selected = Dict{BeliefHyperstate,SelectedTransitionRow}()
        for (s, row) in rows
            s isa AdaptiveState && row isa SelectedTransitionRow ||
                throw(ArgumentError("rows must map AdaptiveStates to SelectedTransitionRows"))
            key = freeze_hyperstate(s)
            haskey(selected, key) && throw(ArgumentError("duplicate parent row"))
            selected[key] = row
        end
        expected = Set(k for k in keys(b._logprobs) if k.outcome == ACTIVE_STATE)
        Set(keys(selected)) == expected ||
            throw(ArgumentError("rows must cover exactly the active belief hypotheses"))
        for (key, row) in selected
            selected_physical_transition(pomdp, thaw_hyperstate(key), Int64(a), row)
        end
        return new(Int64(a), copy(b._logprobs), selected)
    end
end

function validate_belief_selection(pomdp, b, a, rows::BeliefTransitionRows)
    require_belief_action(pomdp, b, a)
    rows.action == a || throw(ArgumentError("selected rows belong to a different action"))
    rows._prior == b._logprobs ||
        throw(ArgumentError("selected rows belong to a different prior belief"))
    return nothing
end

struct ZeroObservationEvidence <: Exception
    action::Int64
    observation::DSObservation
end
Base.showerror(io::IO, e::ZeroObservationEvidence) =
    print(io, "observation ", Tuple(e.observation), " has zero probability under the belief and selected rows for action ", e.action)

# Evaluate the same known, independent-cell camera model in log space. Taking
# log(pdf(SparseCat(...))) would already have lost underflowed likelihoods.
function belief_camera_loglikelihood(pomdp, config, outcome, o)
    outcome != ACTIVE_STATE && return o == TERMINAL_OBSERVATION ? 0.0 : -Inf
    o == TERMINAL_OBSERVATION && return -Inf
    cells = visible_cells(pomdp, config)
    return sum(log(pomdp.camera.confusion[cell_class(pomdp, cells[i]), o[i]]) for i in 1:3)
end

"""
    exact_belief_update(pomdp, b, a, o, selected_rows; return_info=false)

Exact successor enumeration conditional on the explicitly supplied HDR-validated
model. Accumulate b(s)*theta_s[j]*O(o|s',a) for every feasible branch, increment
its own count once, merge identical FULL hyperstates, and normalize. Terminal
absorption preserves counts. Previous states are replaced, not appended.

No argmin, posterior-mean fallback, sampling, pruning, or true-state input is
used. Internal log weights retain very small hypotheses. Zero evidence raises
ZeroObservationEvidence. selected_rows is a BeliefTransitionRows prepared for
this prior/action, or an explicit parent-to-SelectedTransitionRow dictionary.
With return_info=true, return (belief, log_evidence, parents, expanded_branches).
"""
function exact_belief_update(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief,
                             a::Integer, o::DSObservation, rows::BeliefTransitionRows;
                             return_info::Bool=false)
    validate_belief_selection(pomdp, b, a, rows)
    obsindex(pomdp, o)
    masses = Dict{BeliefHyperstate,Float64}()
    likelihoods = Dict{Tuple{DSState,AdaptiveOutcome},Float64}()
    expanded = 0
    for (parent, logw) in b._logprobs
        s = thaw_hyperstate(parent)
        physical = is_terminal(s) ? locomotion_mode_transition(pomdp, s, a, [1.0]) :
            selected_physical_transition(pomdp, s, a, rows._rows[parent])
        for ((rp, mp), probability) in weighted_iterator(physical)
            expanded += 1
            probability == 0 && continue
            destination = adaptive_state(pomdp, rp, mp)
            cache_key = (rp, destination.outcome)
            loglikelihood = get!(likelihoods, cache_key) do
                belief_camera_loglikelihood(pomdp, rp, destination.outcome, o)
            end
            loglikelihood == -Inf && continue
            counts = parameter_update_transition(pomdp, s, a, rp, mp).val
            child = freeze_hyperstate(adaptive_state(pomdp, rp, mp, counts))
            mass = logw + log(probability) + loglikelihood
            masses[child] = belief_logadd(get(masses, child, -Inf), mass)
        end
    end
    isempty(masses) && throw(ZeroObservationEvidence(Int64(a), o))
    log_evidence = belief_logtotal(masses)
    posterior = ExactAdaptiveBelief(masses, Val(:logmass))
    return return_info ? (; belief=posterior, log_evidence, parents=length(b),
                           expanded_branches=expanded) : posterior
end

function exact_belief_update(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief,
                             a::Integer, o::DSObservation, rows::AbstractDict; kwargs...)
    selected = BeliefTransitionRows(pomdp, b, a, rows)
    return exact_belief_update(pomdp, b, a, o, selected; kwargs...)
end

"""
    ExactAdaptiveUpdater(pomdp)

POMDPs adapter for the conditional exact updater. Call prepare_belief_update!
with rows selected for the current belief/action BEFORE observing the outcome.
update consumes that prepared selection on success. Missing, stale, or reused
selections fail explicitly; an error leaves the selection available for diagnosis.
No row selector or simulator state is consulted by this adapter.
"""
mutable struct ExactAdaptiveUpdater <: POMDPs.Updater
    pomdp::DroneSurveillancePOMDP
    _pending::Union{Nothing,BeliefTransitionRows}
end
ExactAdaptiveUpdater(pomdp::DroneSurveillancePOMDP) = ExactAdaptiveUpdater(pomdp, nothing)

function prepare_belief_update!(up::ExactAdaptiveUpdater, b::ExactAdaptiveBelief, a::Integer, rows)
    selected = rows isa BeliefTransitionRows ? rows : BeliefTransitionRows(up.pomdp, b, a, rows)
    validate_belief_selection(up.pomdp, b, a, selected)
    up._pending = selected
    return up
end

function POMDPs.update(up::ExactAdaptiveUpdater, b::ExactAdaptiveBelief, a::Integer, o::DSObservation)
    up._pending === nothing && throw(ArgumentError("prepare explicit transition rows before updating the belief"))
    posterior = exact_belief_update(up.pomdp, b, a, o, up._pending)
    up._pending = nothing
    return posterior
end

function POMDPs.initialize_belief(up::ExactAdaptiveUpdater, s::AdaptiveState)
    b = ExactAdaptiveBelief(up.pomdp, s)
    up._pending = nothing
    return b
end
POMDPs.initialize_belief(up::ExactAdaptiveUpdater, ::Nothing) =
    initialize_belief(up, initial_adaptive_state(up.pomdp))

function POMDPs.initialize_belief(up::ExactAdaptiveUpdater, b::ExactAdaptiveBelief)
    for s in support(b)
        validate_belief_state(up.pomdp, s)
    end
    up._pending = nothing
    return ExactAdaptiveBelief(copy(b._logprobs), Val(:logmass))
end

function POMDPs.initialize_belief(up::ExactAdaptiveUpdater, distribution)
    states = collect(support(distribution))
    all(s -> s isa AdaptiveState, states) ||
        throw(ArgumentError("initial distribution must contain AdaptiveStates, not legacy DSStates"))
    b = ExactAdaptiveBelief(up.pomdp, [s => pdf(distribution, s) for s in states])
    up._pending = nothing
    return b
end
