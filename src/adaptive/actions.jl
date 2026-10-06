# Actions for DroneSurveillanceAdaptive, included inside its module.
# Translations are relative to the drone's heading; rotations happen in place.
const ACTIONS_DICT = Dict(:forward => 1, :backward => 2, :move_left => 3,
                          :move_right => 4, :rotate_clockwise => 5,
                          :rotate_counterclockwise => 6)
const N_ACTIONS = length(ACTIONS_DICT)

POMDPs.actions(pomdp::DroneSurveillancePOMDP) = 1:N_ACTIONS
POMDPs.actions(pomdp::DroneSurveillancePOMDP, s::DSState) = actions(pomdp)
POMDPs.actionindex(pomdp::DroneSurveillancePOMDP, a::Int64) = a
