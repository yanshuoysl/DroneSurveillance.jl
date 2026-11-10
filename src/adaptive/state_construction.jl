# Map-aware hyperstate construction; loaded after the POMDP type.

"""
    adaptive_state(pomdp, config, mode=NOMINAL_MODE, counts=DirichletCounts())

Construct a map-aware augmented state. Obstacle cells are unsafe; region B is a
successful terminal configuration. Outside coordinates are canonicalized to
unsafe, including the executed outcome of an outward command.
This classifies an outcome only; it does not implement movement, fault
dynamics, or count updates.
"""
function adaptive_state(pomdp::DroneSurveillancePOMDP, config::DSState,
                        mode::FaultMode=NOMINAL_MODE,
                        counts::DirichletCounts=DirichletCounts())
    x, y = config.quad
    if !(1 <= x <= pomdp.size[1] && 1 <= y <= pomdp.size[2]) ||
       config.quad in pomdp.obstacles
        return AdaptiveState(UNSAFE_CONFIG, mode, counts; outcome=UNSAFE_STATE)
    end
    outcome = config.quad == pomdp.region_B ? GOAL_STATE : ACTIVE_STATE
    return AdaptiveState(config, mode, counts; outcome=outcome)
end

"""Known initial state: region A, configured initial heading, nominal mode, all-two counts."""
function initial_adaptive_state(pomdp::DroneSurveillancePOMDP)
    s = adaptive_state(pomdp, DSState(pomdp.region_A, pomdp.initial_heading))
    is_unsafe(s) && throw(ArgumentError("initial position must be a free grid cell"))
    return s
end


POMDPs.initialstate(pomdp::DroneSurveillancePOMDP) =
    Deterministic(initial_adaptive_state(pomdp))
