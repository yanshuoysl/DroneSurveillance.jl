# Augmented state s = (r, m, φ), with robot configuration r = (x, y, ψ).
# Model-independent hyperstate types; loaded after DSState, before the POMDP type.

"""
Six fault flags, ordered forward, backward, left, right, clockwise rotation,
counterclockwise rotation. `true` marks the corresponding action's fault flag.
The learned transition support retains both staying and commanded motion for
every admissible next mode; a flag does not deterministically select the outcome.
"""
const FaultMode = NTuple{6, Bool}
const NOMINAL_MODE = ntuple(_ -> false, 6)
const N_FAULT_MODES = 2^length(NOMINAL_MODE)

is_nominal(mode::FaultMode) = mode == NOMINAL_MODE

"""Whether the action's fault flag is set; IDs follow `adaptive/actions.jl`."""
function is_failed(mode::FaultMode, action::Integer)
    1 <= action <= length(mode) || throw(ArgumentError("action must be from 1 to 6"))
    return mode[action]
end

"""One-based mode index, with the forward fault as the least significant bit."""
mode_index(mode::FaultMode) = 1 + sum(Int(mode[i]) << (i - 1) for i in 1:6)

function mode_from_index(index::Integer)
    1 <= index <= N_FAULT_MODES || throw(ArgumentError("mode index must be from 1 to 64"))
    mask = Int(index) - 1
    return ntuple(i -> !iszero(mask & (1 << (i - 1))), 6)
end

all_fault_modes() = [mode_from_index(i) for i in 1:N_FAULT_MODES]

"""Membership in E: m ≼ m′ and ‖m′ - m‖₁ ≤ 1 (no fault recovery)."""
function is_successor_mode(mode::FaultMode, next_mode::FaultMode)
    all(i -> !mode[i] || next_mode[i], 1:6) || return false
    return count(identity, next_mode) - count(identity, mode) <= 1
end

"""𝓜(m): the current mode, followed by each mode with one additional fault."""
function successor_modes(mode::FaultMode)
    successors = FaultMode[mode]
    for i in 1:6
        mode[i] && continue
        push!(successors, ntuple(j -> mode[j] || j == i, 6))
    end
    return successors
end

@enum AdaptiveOutcome ACTIVE_STATE GOAL_STATE UNSAFE_STATE

# Executed boundary crossings and obstacle entries share this unsafe configuration.
# Goal configurations keep their actual grid position and heading.
const UNSAFE_CONFIG = DSState((-1, -1), 0)

is_grid_config(config::DSState) = all(>(0), config.quad)

"""
    CountKey(config, mode, action, next_config, next_mode)

Index of φ^a_{(r,m),(r′,m′)}. Robot configurations r and r′ include position and heading.
Unsafe destinations use `UNSAFE_CONFIG`; successful destinations use the actual
goal configuration. Counts and outcome labels are not part of the index.

This low-level constructor checks action IDs and irreversible mode evolution.
Use `CountKey(pomdp, ...)` to additionally validate map- and action-dependent
support. This storage constructor alone does not establish physical feasibility.
"""
struct CountKey
    config::DSState
    mode::FaultMode
    action::Int
    next_config::DSState
    next_mode::FaultMode

    function CountKey(config::DSState, mode::FaultMode, action::Integer,
                      next_config::DSState, next_mode::FaultMode)
        is_grid_config(config) || throw(ArgumentError("source must be a grid configuration"))
        (is_grid_config(next_config) || next_config == UNSAFE_CONFIG) ||
            throw(ArgumentError("destination must be a grid configuration or UNSAFE_CONFIG"))
        1 <= action <= 6 || throw(ArgumentError("action must be from 1 to 6"))
        is_successor_mode(mode, next_mode) ||
            throw(ArgumentError("mode transition cannot recover faults or add multiple faults"))
        return new(config, mode, Int(action), next_config, next_mode)
    end
end

const DIRICHLET_PRIOR_COUNT = 2

"""
    DirichletCounts()

Sparse representation of the full Dirichlet count vector φ, initially all twos
on feasible transition entries. `increments[key]` stores φ[key] - 2; missing
entries have count 2. Thus every learned-row exponent φ[key] - 1 stays positive.
Impossible entries are excluded by transition support, not assigned count 2
merely because they are absent from this dictionary.

Treat `increments` as read-only. Use `increment_count` to obtain a new snapshot;
each belief history can then retain its own counts. The constructor copies its
input dictionary so later changes to that input do not alter this snapshot.
"""
struct DirichletCounts
    increments::Dict{CountKey, Int}

    function DirichletCounts(increments::AbstractDict{CountKey, <:Integer})
        all(n -> n >= 0, values(increments)) ||
            throw(ArgumentError("count increments must be nonnegative"))
        return new(Dict{CountKey, Int}(key => Int(n) for (key, n) in increments if n != 0))
    end
end

DirichletCounts() = DirichletCounts(Dict{CountKey, Int}())

"""Read a storage entry (implicit prior 2); use `count_value(pomdp, phi, key)` to validate feasibility."""
count_value(phi::DirichletCounts, key::CountKey) =
    Base.Checked.checked_add(DIRICHLET_PRIOR_COUNT, get(phi.increments, key, 0))

"""Increment a storage key without mutation; use `increment_count(pomdp, phi, key)` to validate support."""
function increment_count(phi::DirichletCounts, key::CountKey)
    updated = DirichletCounts(phi.increments)
    increment = Base.Checked.checked_add(get(updated.increments, key, 0), 1)
    # Leave room for the implicit prior count of 2.
    Base.Checked.checked_add(increment, DIRICHLET_PRIOR_COUNT)
    updated.increments[key] = increment
    return updated
end

"""
    count_row(phi, config, mode, action, destinations)

Return a row's Dirichlet parameter vector in the supplied destination order.
Each destination `(next_config, next_mode)` represents a pair (r′, m′) in
S(r, m, a). This low-level overload trusts the supplied destinations; use
`count_row(pomdp, phi, config, mode, action)` for the complete validated support.
No full transition table or count-state space is enumerated.
"""
function count_row(phi::DirichletCounts, config::DSState, mode::FaultMode,
                   action::Integer, destinations)
    keys = [CountKey(config, mode, action, next_config, next_mode)
            for (next_config, next_mode) in destinations]
    isempty(keys) && throw(ArgumentError("a Dirichlet row needs at least one feasible destination"))
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("a Dirichlet row cannot contain duplicate destinations"))
    return [count_value(phi, key) for key in keys]
end

# Compare snapshots by their contents, not by dictionary identity.
Base.:(==)(a::DirichletCounts, b::DirichletCounts) = a.increments == b.increments
Base.isequal(a::DirichletCounts, b::DirichletCounts) = isequal(a.increments, b.increments)
Base.hash(phi::DirichletCounts, h::UInt) = hash(phi.increments, h)

"""
    AdaptiveState(config, mode=NOMINAL_MODE, counts=DirichletCounts();
                  outcome=ACTIVE_STATE)

Augmented state s = (r, m, φ). `config` is r = (x, y, ψ), `mode` is m, and `counts`
represents φ. `outcome` distinguishes active, successful terminal, and unsafe terminal
states; it is derived from the configuration and map by `adaptive_state`.

Terminal states retain their mode and counts. The separate parameter-update
factor and combined kernels increment the entry count once and preserve counts
on subsequent absorbing transitions.
There is no finite enumeration of all count vectors.
"""
struct AdaptiveState
    config::DSState
    mode::FaultMode
    counts::DirichletCounts
    outcome::AdaptiveOutcome

    function AdaptiveState(config::DSState, mode::FaultMode=NOMINAL_MODE,
                           counts::DirichletCounts=DirichletCounts();
                           outcome::AdaptiveOutcome=ACTIVE_STATE)
        if outcome == UNSAFE_STATE
            config == UNSAFE_CONFIG ||
                throw(ArgumentError("unsafe states must use UNSAFE_CONFIG"))
        else
            is_grid_config(config) ||
                throw(ArgumentError("active and goal states must use grid configurations"))
        end
        return new(config, mode, counts, outcome)
    end
end

is_unsafe(s::AdaptiveState) = s.outcome == UNSAFE_STATE
is_goal(s::AdaptiveState) = s.outcome == GOAL_STATE
is_terminal(s::AdaptiveState) = s.outcome != ACTIVE_STATE

Base.:(==)(a::AdaptiveState, b::AdaptiveState) =
    a.config == b.config && a.mode == b.mode && a.counts == b.counts && a.outcome == b.outcome
Base.isequal(a::AdaptiveState, b::AdaptiveState) =
    isequal(a.config, b.config) && isequal(a.mode, b.mode) &&
    isequal(a.counts, b.counts) && isequal(a.outcome, b.outcome)
Base.hash(s::AdaptiveState, h::UInt) = hash((s.config, s.mode, s.counts, s.outcome), h)
