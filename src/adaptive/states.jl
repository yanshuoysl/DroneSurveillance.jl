# State space for DroneSurveillanceAdaptive, included inside its module.
# Four headings per traversable grid cell, plus the terminal sentinel.
function free_positions(pomdp::DroneSurveillancePOMDP)
    return [DSPos(xy[1], xy[2]) for xy in vec(collect(CartesianIndices(pomdp.size)))
            if DSPos(xy[1], xy[2]) ∉ pomdp.obstacles]
end

POMDPs.states(pomdp::DroneSurveillancePOMDP) = pomdp
Base.length(pomdp::DroneSurveillancePOMDP) =
    (prod(pomdp.size) - length(pomdp.obstacles)) * length(HEADINGS) + 1

function POMDPs.stateindex(pomdp::DroneSurveillancePOMDP, s::DSState)
    isterminal(pomdp, s) && return length(pomdp)
    positions = free_positions(pomdp)
    i = findfirst(==(s.quad), positions)
    i === nothing && throw(ArgumentError("state position must be a free grid cell"))
    return i + div(s.heading, 90) * length(positions)
end

function state_from_index(pomdp::DroneSurveillancePOMDP, i::Int64)
    i == length(pomdp) && return pomdp.terminal_state
    1 <= i < length(pomdp) || throw(BoundsError(pomdp, i))
    positions = free_positions(pomdp)
    heading_index, position_index = divrem(i - 1, length(positions))
    return DSState(positions[position_index + 1], HEADINGS[heading_index + 1])
end

function Base.iterate(pomdp::DroneSurveillancePOMDP, i::Int64=1)
    i > length(pomdp) && return nothing
    return (state_from_index(pomdp, i), i + 1)
end

POMDPs.initialstate(pomdp::DroneSurveillancePOMDP) =
    Deterministic(DSState(pomdp.region_A, pomdp.initial_heading))
