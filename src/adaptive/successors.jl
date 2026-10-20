# Deterministic support only: no probabilities, count updates, or sampling.

function commanded_config(config::DSState, action::Integer)
    if action == ACTIONS_DICT[:rotate_clockwise]
        return DSState(config.quad, mod(config.heading + 90, 360))
    elseif action == ACTIONS_DICT[:rotate_counterclockwise]
        return DSState(config.quad, mod(config.heading - 90, 360))
    end
    forward = HEADING_DIRS[div(config.heading, 90) + 1]
    direction = if action == ACTIONS_DICT[:forward]
        forward
    elseif action == ACTIONS_DICT[:backward]
        -forward
    elseif action == ACTIONS_DICT[:move_left]
        DSPos(-forward[2], forward[1])
    else # move right; public callers validate the action ID first.
        DSPos(forward[2], -forward[1])
    end
    return DSState(config.quad + direction, config.heading)
end

function support_source(pomdp::DroneSurveillancePOMDP, config::DSState,
                        mode::FaultMode, action::Integer)
    1 <= action <= N_ACTIONS || throw(ArgumentError("action must be from 1 to $N_ACTIONS"))
    source = adaptive_state(pomdp, config, mode)
    config == source.config ||
        throw(ArgumentError("source must be a free grid configuration or UNSAFE_CONFIG"))
    if !is_terminal(source) && !command_in_bounds(pomdp, config, action)
        throw(ArgumentError("action would move outside the grid boundary"))
    end
    return source
end

function successor_configs(pomdp::DroneSurveillancePOMDP, config::DSState,
                           action::Integer)
    commanded = adaptive_state(pomdp, commanded_config(config, action)).config
    return unique([config, commanded])
end

"""
    feasible_successors(pomdp, config, mode, action)

Return the complete ordered support S(r,m,a) as `(next_config, next_mode)` pairs
for all six relative-motion actions. For an active source this is the Cartesian
product `{stay, commanded configuration} × successor_modes(mode)`. Every next
mode is paired with BOTH motion outcomes, including modes whose commanded-action
fault flag is set. The flags do not deterministically select a motion outcome;
transition probabilities over the joint pairs remain to be learned. Cartesian
support does not imply probabilistic independence of configuration and mode.
Modes stay unchanged or gain one fault, without recovery. No sideways slip,
extra movement, or unintended rotation is included.

Outward boundary commands are invalid and raise ArgumentError: they have no
learned row, including no stay branch. Obstacle contact maps to `UNSAFE_CONFIG`.
Goal and unsafe sources are absorbing, returning
only `(config, mode)` without further degradation. Raw obstacle/outside source
coordinates are rejected; unsafe sources must already be canonical.

Ordering is configuration-major: all next modes for staying first, followed by
all next modes for the commanded configuration. Within each block, the unchanged
mode comes first, then one new fault in action-ID order. Canonically identical
configurations are deduplicated. Support is independent of counts/probabilities.
The legacy DSState simulator still treats obstacle contact as blocked motion.
"""
function feasible_successors(pomdp::DroneSurveillancePOMDP, config::DSState,
                             mode::FaultMode, action::Integer)
    source = support_source(pomdp, config, mode, action)
    is_terminal(source) && return [(config, mode)]
    return [(next_config, next_mode)
            for next_config in successor_configs(pomdp, config, action)
            for next_mode in successor_modes(mode)]
end

"""
    is_feasible_successor(pomdp, config, mode, action, next_config, next_mode)

Test membership in S(r,m,a). Invalid action IDs, noncanonical source/destination
configurations, fault recovery, and multiple new faults return false.
"""
function is_feasible_successor(pomdp::DroneSurveillancePOMDP, config::DSState,
                               mode::FaultMode, action::Integer,
                               next_config::DSState, next_mode::FaultMode)
    1 <= action <= N_ACTIONS || return false
    source = adaptive_state(pomdp, config, mode)
    config == source.config || return false
    if is_terminal(source)
        return next_config == config && next_mode == mode
    end
    command_in_bounds(pomdp, config, action) || return false
    is_successor_mode(mode, next_mode) || return false
    return next_config in successor_configs(pomdp, config, action)
end

function require_learned_row(pomdp::DroneSurveillancePOMDP, config::DSState,
                             mode::FaultMode, action::Integer)
    source = support_source(pomdp, config, mode, action)
    is_terminal(source) && throw(ArgumentError("terminal sources have no learned Dirichlet row"))
    return nothing
end

"""
    CountKey(pomdp, config, mode, action, next_config, next_mode)

Construct a map- and action-validated count key. Impossible transitions are
structural zeros, excluded from the Dirichlet row, and raise `ArgumentError`.
Terminal sources have no learned row. The constructor without `pomdp` remains
a low-level storage primitive for manually supplied/illustrative counts.
"""
function CountKey(pomdp::DroneSurveillancePOMDP, config::DSState,
                  mode::FaultMode, action::Integer,
                  next_config::DSState, next_mode::FaultMode)
    require_learned_row(pomdp, config, mode, action)
    is_feasible_successor(pomdp, config, mode, action, next_config, next_mode) ||
        throw(ArgumentError("destination is outside S(r,m,a) for this action and fault mode"))
    return CountKey(config, mode, action, next_config, next_mode)
end

function validate_count_key(pomdp::DroneSurveillancePOMDP, key::CountKey)
    CountKey(pomdp, key.config, key.mode, key.action, key.next_config, key.next_mode)
    return nothing
end

"""Read a count after validating its key against S(r,m,a); impossible keys raise an error."""
function count_value(pomdp::DroneSurveillancePOMDP, phi::DirichletCounts, key::CountKey)
    validate_count_key(pomdp, key)
    return count_value(phi, key)
end

"""Increment a feasible count in a new snapshot; reject impossible keys without mutation."""
function increment_count(pomdp::DroneSurveillancePOMDP, phi::DirichletCounts, key::CountKey)
    validate_count_key(pomdp, key)
    return increment_count(phi, key)
end

"""
    count_row(pomdp, phi, config, mode, action[, destinations])

Read a complete feasible Dirichlet row. Destinations default to S(r,m,a).
An explicitly supplied ordering must contain every feasible pair exactly once.
Infeasible or incomplete rows are rejected. Use this map-aware overload for
model counts; the overload without `pomdp` only reads supplied storage keys.
"""
function count_row(pomdp::DroneSurveillancePOMDP, phi::DirichletCounts,
                   config::DSState, mode::FaultMode, action::Integer,
                   destinations=feasible_successors(pomdp, config, mode, action))
    require_learned_row(pomdp, config, mode, action)
    expected = feasible_successors(pomdp, config, mode, action)
    ordered = collect(destinations)
    length(ordered) == length(expected) && Set(ordered) == Set(expected) ||
        throw(ArgumentError("destinations must contain every pair in S(r,m,a) exactly once"))
    return count_row(phi, config, mode, action, ordered)
end
