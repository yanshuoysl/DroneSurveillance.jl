module DroneSurveillanceAdaptive

using POMDPs
using POMDPTools
using Parameters
using StaticArrays
using Compose
using Colors
using Random

export DSPos, DSState, DSObservation, DroneSurveillancePOMDP, ACTIONS_DICT,
       KnownCamera, OBS_CLASSES, observation_labels, visible_cells, random_obstacles

const DSPos = SVector{2, Int64}
const HEADINGS = (0, 90, 180, 270) # north, east, south, west; clockwise degrees
const HEADING_DIRS = (DSPos(0, 1), DSPos(1, 0), DSPos(0, -1), DSPos(-1, 0))

"""
    DSState(quad, heading=0)

Drone position and heading in clockwise degrees from north: 0, 90, 180, or 270.
"""
struct DSState
    quad::DSPos
    heading::Int64

    function DSState(quad, heading::Integer=0)
        heading in HEADINGS || throw(ArgumentError("heading must be 0, 90, 180, or 270 degrees"))
        return new(DSPos(quad), Int64(heading))
    end
end

include("adaptive/camera.jl")
include("adaptive/obstacles.jl")

"""
    DroneSurveillancePOMDP(; size=(5, 5), region_A=[1, 1],
        region_B=[size[1], size[2]], initial_heading=0,
        obstacles=random_obstacles(size; exclude=[region_A, region_B]),
        camera=KnownCamera(), discount_factor=0.95)

Drone surveillance variant without a ground agent. The drone starts at region A,
observes noisy classes for the three cells immediately ahead, and earns +1 for
occupying region B. Its position and heading are not part of the camera output. Movement
is relative to the heading; rotations change the heading in place. Leaving the grid
enters an absorbing terminal state. Two obstacles are randomly placed by default,
excluding the start and goal. Moving into an obstacle leaves the state unchanged.
The field of view is shown in the rendering.
"""
@with_kw mutable struct DroneSurveillancePOMDP <: POMDP{DSState, Int64, DSObservation}
    size::Tuple{Int64, Int64} = (5, 5)
    region_A::DSPos = [1, 1]
    region_B::DSPos = [size[1], size[2]]
    initial_heading::Int64 = 0
    obstacles::Vector{DSPos} = random_obstacles(size; exclude=[region_A, region_B])
    @assert length(unique(obstacles)) == length(obstacles) "obstacle cells must be distinct"
    @assert all(pos -> 1 <= pos[1] <= size[1] && 1 <= pos[2] <= size[2], obstacles) "obstacles must be inside the grid"
    @assert region_A ∉ obstacles && region_B ∉ obstacles "start and goal must be free"
    camera::KnownCamera = KnownCamera()
    terminal_state::DSState = DSState([-1, -1])
    discount_factor::Float64 = 0.95
end

POMDPs.isterminal(pomdp::DroneSurveillancePOMDP, s::DSState) = s == pomdp.terminal_state
POMDPs.discount(pomdp::DroneSurveillancePOMDP) = pomdp.discount_factor
POMDPs.reward(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64) =
    !isterminal(pomdp, s) && s.quad == pomdp.region_B ? 1.0 : 0.0

include("adaptive/states.jl")
include("adaptive/actions.jl")
include("adaptive/observation.jl")

function POMDPs.transition(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64)
    1 <= a <= N_ACTIONS || throw(ArgumentError("action must be an integer from 1 to $N_ACTIONS"))
    isterminal(pomdp, s) && return Deterministic(pomdp.terminal_state)
    if a == ACTIONS_DICT[:rotate_clockwise]
        return Deterministic(DSState(s.quad, mod(s.heading + 90, 360)))
    elseif a == ACTIONS_DICT[:rotate_counterclockwise]
        return Deterministic(DSState(s.quad, mod(s.heading - 90, 360)))
    end

    forward = HEADING_DIRS[div(s.heading, 90) + 1]
    direction = if a == ACTIONS_DICT[:forward]
        forward
    elseif a == ACTIONS_DICT[:backward]
        -forward
    elseif a == ACTIONS_DICT[:move_left]
        DSPos(-forward[2], forward[1])
    else # move right
        DSPos(forward[2], -forward[1])
    end
    quad = s.quad + direction
    if !(1 <= quad[1] <= pomdp.size[1] && 1 <= quad[2] <= pomdp.size[2])
        return Deterministic(pomdp.terminal_state)
    end
    quad in pomdp.obstacles && return Deterministic(s)
    return Deterministic(DSState(quad, s.heading))
end

function POMDPTools.render(pomdp::DroneSurveillancePOMDP, step)
    nx, ny = pomdp.size
    s = get(step, :sp, get(step, :s, nothing))
    cells = []
    for x in 1:nx, y in 1:ny
        pos = DSPos(x, y)
        clr = "white"
        if s !== nothing && in_fov(pomdp, s, pos)
            clr = ARGB(0.0, 0.0, 1.0, 0.9)
        end
        is_survey = pos == pomdp.region_A || pos == pomdp.region_B
        is_survey && (clr = "green")
        pos in pomdp.obstacles && (clr = "orange")
        push!(cells, compose(cell_ctx(pos, pomdp.size), rectangle(), fill(clr)))
    end
    grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), fill(nothing), stroke("black"))
    quad = s === nothing || isterminal(pomdp, s) ? nothing :
        render_quad(cell_ctx(s.quad, pomdp.size), s.heading)
    sz = min(w, h)
    scene = compose(context((w - sz)/2, (h - sz)/2, sz, sz), quad, grid, outline)
    background = compose(context(), rectangle(), fill("white"), stroke(nothing))
    return compose(context(), scene, background)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1)/nx, (ny - y)/ny, 1/nx, 1/ny)
end

function render_quad(ctx, heading)
    center = compose(context(), ellipse(0.5, 0.5, 0.20, 0.3), fill("orange"), stroke("black"))
    rotors = [compose(context(), circle(x, y, 0.17), fill("gray"), stroke("black"))
              for (x, y) in ((0.2, 0.8), (0.8, 0.8), (0.2, 0.2), (0.8, 0.2))]
    # A yellow triangle marks the nose; screen y increases downwards.
    fx, fy = HEADING_DIRS[div(heading, 90) + 1]
    dx, dy = fx, -fy
    tip = (0.5 + 0.28dx, 0.5 + 0.28dy)
    base_left = (0.5 + 0.05dx - 0.10dy, 0.5 + 0.05dy + 0.10dx)
    base_right = (0.5 + 0.05dx + 0.10dy, 0.5 + 0.05dy - 0.10dx)
    nose = compose(context(), polygon([tip, base_left, base_right]), fill("yellow"), stroke("black"))
    return compose(ctx, nose, rotors..., center)
end

end
