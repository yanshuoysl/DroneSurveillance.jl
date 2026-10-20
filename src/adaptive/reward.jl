"""
    adaptive_reward(pomdp, s::AdaptiveState, action, sp::AdaptiveState)
    reward(pomdp, s::AdaptiveState, action, sp::AdaptiveState)

Transition reward for the AR-POMDP: +100 on entry into the goal, -100 for an
actual obstacle collision, and -1 otherwise (including stay, rotation, and
ordinary movement). These are alternative rewards, not bonuses added to a
step cost. An already terminal source has reward 0 on its absorbing transition.

Feasible unsafe entries are obstacle collisions and receive -100. Outward
boundary commands are invalid and rejected before reward evaluation. Staying
before an obstacle is not a collision and receives -1.

Sources and destinations must be canonical, have consistent outcome labels,
and form a feasible configuration/mode transition. The numeric reward does
not use counts, their increment, theta, or the HDR. Counts are never changed.
The legacy DSState reward remains a separate interface.
"""
function adaptive_reward(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                          action::Integer, sp::AdaptiveState)
    source = transition_source(pomdp, s, action)
    destination = adaptive_state(pomdp, sp.config, sp.mode)
    sp.config == destination.config && sp.outcome == destination.outcome ||
        throw(ArgumentError("destination configuration or outcome is inconsistent with the map"))
    is_feasible_successor(pomdp, s.config, s.mode, action, sp.config, sp.mode) ||
        throw(ArgumentError("reward requires a feasible configuration/mode transition"))
    is_terminal(source) && return 0.0
    is_goal(destination) && return 100.0
    is_unsafe(destination) && return -100.0
    return -1.0
end

POMDPs.reward(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
              action::Integer, sp::AdaptiveState) = adaptive_reward(pomdp, s, action, sp)
