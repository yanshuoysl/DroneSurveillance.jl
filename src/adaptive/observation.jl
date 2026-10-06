# Camera observes one row of three cells ahead of the drone.
"""Return front-left, front-center, front-right grid coordinates, including out-of-grid cells."""
function visible_cells(pomdp::DroneSurveillancePOMDP, s::DSState)
    isterminal(pomdp, s) && return ()
    forward = HEADING_DIRS[div(s.heading, 90) + 1]
    left = DSPos(-forward[2], forward[1])
    center = s.quad + forward
    return (center + left, center, center - left)
end

in_fov(pomdp::DroneSurveillancePOMDP, s::DSState, pos::DSPos) = pos in visible_cells(pomdp, s)

function cell_class(pomdp::DroneSurveillancePOMDP, pos::DSPos)
    if !(1 <= pos[1] <= pomdp.size[1] && 1 <= pos[2] <= pomdp.size[2])
        return 3 # outside
    elseif pos == pomdp.region_A || pos == pomdp.region_B
        return 2 # survey
    elseif pos in pomdp.obstacles
        return 4 # obstacle
    else
        return 1 # ordinary
    end
end

const OBS_DIMENSIONS = (N_CELL_CLASSES, N_CELL_CLASSES, N_CELL_CLASSES)
const CAMERA_OBSERVATIONS = vec([DSObservation(Tuple(i)) for i in CartesianIndices(OBS_DIMENSIONS)])
const ALL_OBSERVATIONS = vcat(CAMERA_OBSERVATIONS, [TERMINAL_OBSERVATION])

POMDPs.observations(pomdp::DroneSurveillancePOMDP) = ALL_OBSERVATIONS

function POMDPs.obsindex(pomdp::DroneSurveillancePOMDP, o::DSObservation)
    o == TERMINAL_OBSERVATION && return length(ALL_OBSERVATIONS)
    all(i -> 1 <= i <= N_CELL_CLASSES, o) || throw(ArgumentError("invalid camera observation"))
    return LinearIndices(OBS_DIMENSIONS)[o[1], o[2], o[3]]
end

function POMDPs.observation(pomdp::DroneSurveillancePOMDP, a::Int64, sp::DSState)
    isterminal(pomdp, sp) && return Deterministic(TERMINAL_OBSERVATION)
    truth = map(pos -> cell_class(pomdp, pos), visible_cells(pomdp, sp))
    model = pomdp.camera.confusion
    probs = [prod(model[truth[i], o[i]] for i in 1:3) for o in CAMERA_OBSERVATIONS]
    return SparseCat(CAMERA_OBSERVATIONS, probs)
end
