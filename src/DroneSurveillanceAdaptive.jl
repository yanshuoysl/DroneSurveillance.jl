module DroneSurveillanceAdaptive

using POMDPs
using POMDPTools
using Parameters
using StaticArrays
using Compose
using Colors
using Random
using Distributions: Dirichlet, pdf
using Statistics: quantile

export DSPos, DSState, DSObservation, DroneSurveillancePOMDP, ACTIONS_DICT,
       KnownCamera, OBS_CLASSES, observation_labels, visible_cells, random_obstacles,
       AdaptiveState, AdaptiveOutcome, ACTIVE_STATE, GOAL_STATE, UNSAFE_STATE,
       FaultMode, NOMINAL_MODE, N_FAULT_MODES, UNSAFE_CONFIG,
       is_nominal, is_failed, mode_index, mode_from_index, all_fault_modes,
       is_successor_mode, successor_modes, CountKey, DirichletCounts,
       feasible_successors, is_feasible_successor,
       count_value, increment_count, count_row,
       locomotion_mode_uncertainty, locomotion_mode_transition,
       parameter_update_transition,
       adaptive_reward,
       posterior_predictive_row, adaptive_transition, robust_transition,
       adaptive_state, initial_adaptive_state, is_unsafe, is_goal, is_terminal,
       termination_reason, simulate_adaptive,
       sample_fault_schedule,
       DirichletHDR, estimate_tau, build_row_hdr, contains

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
is relative to the heading; rotations change the heading in place. Commands that
would leave the grid are invalid and excluded from state-dependent action sets.
Two obstacles are randomly placed by default,
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

include("adaptive/hyperstates.jl")

POMDPs.isterminal(pomdp::DroneSurveillancePOMDP, s::DSState) = s == pomdp.terminal_state
POMDPs.isterminal(pomdp::DroneSurveillancePOMDP, s::AdaptiveState) = is_terminal(s)
POMDPs.discount(pomdp::DroneSurveillancePOMDP) = pomdp.discount_factor
POMDPs.reward(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64) =
    !isterminal(pomdp, s) && s.quad == pomdp.region_B ? 1.0 : 0.0

include("adaptive/states.jl")
include("adaptive/actions.jl")
include("adaptive/faults.jl")
include("adaptive/successors.jl")

# Boundary feasibility depends only on configuration, never on hidden fault flags.
function command_in_bounds(pomdp::DroneSurveillancePOMDP, config::DSState, a::Integer)
    1 <= a <= N_ACTIONS || return false
    x, y = commanded_config(config, a).quad
    return 1 <= x <= pomdp.size[1] && 1 <= y <= pomdp.size[2]
end

"""
    actions(pomdp, s::Union{DSState,AdaptiveState})

Return commands whose intended destination is inside the grid. All six action
IDs remain in `actions(pomdp)` for indexing. An outward command is unavailable
even if its uncertain outcome could be staying put. Rotations remain available
at every active cell. Obstacle-directed commands remain available; collision is
handled by the transition model. No fault mode or count is used in this check.
Terminal states have no available actions.
"""
function POMDPs.actions(pomdp::DroneSurveillancePOMDP, s::DSState)
    isterminal(pomdp, s) && return Int64[]
    x, y = s.quad
    1 <= x <= pomdp.size[1] && 1 <= y <= pomdp.size[2] ||
        throw(ArgumentError("source configuration must be inside the grid"))
    return Int64[a for a in actions(pomdp) if command_in_bounds(pomdp, s, a)]
end

function POMDPs.actions(pomdp::DroneSurveillancePOMDP, s::AdaptiveState)
    source = adaptive_state(pomdp, s.config, s.mode)
    s.config == source.config && s.outcome == source.outcome ||
        throw(ArgumentError("source configuration or outcome is inconsistent with the map"))
    is_terminal(source) && return Int64[]
    return actions(pomdp, s.config)
end

"""
    actions(pomdp, b::DiscreteBelief)

Commands valid for every configuration with positive belief probability.
This permits boundary-safe selection without revealing the true configuration
or fault mode. Terminal hypotheses do not restrict commands for active ones;
an entirely terminal belief has no available actions.
"""
function POMDPs.actions(pomdp::DroneSurveillancePOMDP, b::DiscreteBelief)
    available = collect(actions(pomdp))
    has_active = false
    for s in support(b)
        pdf(b, s) > 0 || continue
        isterminal(pomdp, s) && continue
        has_active = true
        intersect!(available, actions(pomdp, s))
    end
    return has_active ? available : Int64[]
end

include("adaptive/hdr.jl")
include("adaptive/locomotion.jl")
include("adaptive/parameter_update.jl")
include("adaptive/transition.jl")
include("adaptive/reward.jl")
include("adaptive/observation.jl")

"""Reuse the camera for active hyperstates; goal/collision states emit the terminal observation."""
function POMDPs.observation(pomdp::DroneSurveillancePOMDP, a::Int64, sp::AdaptiveState)
    isterminal(pomdp, sp) && return Deterministic(TERMINAL_OBSERVATION)
    return observation(pomdp, a, sp.config)
end

"""
    termination_reason(pomdp, s::AdaptiveState, steps, H)

Return `:goal`, `:collision`, `:horizon`, or `nothing` while execution may
continue. `steps` counts completed actions, including stays and rotations.
Goal/collision take precedence over the horizon when they occur on action H.
H is a simulation budget, not part of the hyperstate or Dirichlet count key.
"""
function termination_reason(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                            steps::Integer, H::Integer)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    H >= 0 || throw(ArgumentError("H must be nonnegative"))
    source = adaptive_state(pomdp, s.config, s.mode)
    s.config == source.config && s.outcome == source.outcome ||
        throw(ArgumentError("state configuration or outcome is inconsistent with the map"))
    is_goal(s) && return :goal
    is_unsafe(s) && return :collision
    steps >= H && return :horizon
    return nothing
end

"""
    simulate_adaptive(pomdp, policy, belief_updater, initial_belief;
                      H, transition_kernel,
                      initial_state=initial_adaptive_state(pomdp),
                      rng=Random.default_rng())

Run the augmented drone example until goal, collision, or H completed actions.
The required callable `transition_kernel(pomdp, s, a)` returns a distribution
over complete next AdaptiveStates, including the parameter update. For HDR
constraints, supply a callable that selects theta and calls robust_transition.
There is no implicit posterior-mean or worst-case row selection.

Only the environment kernel receives the latent hyperstate. The policy receives
the supplied belief; `update(belief_updater, b, a, o)` receives its camera
observation, never the sampled true mode or counts. The caller supplies a belief
that is already initialized. A scripted PlaybackPolicy may use NothingUpdater
and `initial_belief=nothing`; this does not implement Bayesian belief learning.
The policy must respect boundary availability; invalid commands raise an error
before the transition kernel is called.

Every executed transition records (t,s,a,sp,o,r,b,bp) in a POMDPTools.SimHistory.
The final reward, observation, belief update, and count update are retained.
Initial terminal states and H=0 execute no actions. Returned fields are history,
final_state, final_belief, reason, steps, terminated (goal/collision), and
truncated (horizon). The history contains hidden states for evaluation only.
"""
function simulate_adaptive(pomdp::DroneSurveillancePOMDP, policy, belief_updater,
                           initial_belief; H::Integer, transition_kernel,
                           initial_state::AdaptiveState=initial_adaptive_state(pomdp),
                           rng::AbstractRNG=Random.default_rng())
    s, b = initial_state, initial_belief
    steps = 0
    records = NamedTuple[]
    reason = termination_reason(pomdp, s, steps, H)
    while reason === nothing
        a = action(policy, b)
        a in actions(pomdp, s) ||
            throw(ArgumentError("policy selected an unavailable action: $a"))
        a = Int64(a)
        sp = rand(rng, transition_kernel(pomdp, s, a))
        r = adaptive_reward(pomdp, s, a, sp)
        o = rand(rng, observation(pomdp, a, sp))
        bp = update(belief_updater, b, a, o)
        steps += 1
        push!(records, (; t=steps, s, a, sp, o, r, b, bp))
        s, b = sp, bp
        reason = termination_reason(pomdp, s, steps, H)
    end
    history = POMDPTools.SimHistory(records, discount(pomdp), nothing, nothing)
    return (; history, final_state=s, final_belief=b, reason, steps,
            terminated=reason in (:goal, :collision), truncated=reason == :horizon)
end

function POMDPs.transition(pomdp::DroneSurveillancePOMDP, s::DSState, a::Int64)
    1 <= a <= N_ACTIONS || throw(ArgumentError("action must be an integer from 1 to $N_ACTIONS"))
    isterminal(pomdp, s) && return Deterministic(pomdp.terminal_state)
    command_in_bounds(pomdp, s, a) ||
        throw(ArgumentError("action would move outside the grid boundary"))
    attempted = commanded_config(s, a)
    quad = attempted.quad
    quad in pomdp.obstacles && return Deterministic(s)
    return Deterministic(attempted)
end

function POMDPTools.render(pomdp::DroneSurveillancePOMDP, step)
    nx, ny = pomdp.size
    s = get(step, :sp, get(step, :s, nothing))
    if s isa AdaptiveState
        # Keep the drone visible at the goal; unsafe states have no grid pose.
        s = is_unsafe(s) ? nothing : s.config
    end
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
