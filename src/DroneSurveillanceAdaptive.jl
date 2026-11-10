module DroneSurveillanceAdaptive

using POMDPs
using POMDPTools
using Parameters
using StaticArrays
using Compose
using Colors
using Random
using Distributions: Dirichlet, pdf
import Distributions
using Statistics: quantile
using LinearAlgebra

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
       sample_fault_schedule, HiddenFaultEnvironment, physical_step,
       simulate_robust, save_reward_chart,
       ExactAdaptiveBelief, ExactAdaptiveUpdater, SelectedTransitionRow,
       BeliefTransitionRows, exact_belief_update, prepare_belief_update!,
       ZeroObservationEvidence,
       DirichletHDR, estimate_tau, build_row_hdr, contains,
       HDRRowMinimum, RowMinimizationError, minimize_hdr_row,
       RobustPlannerOptions, RobustValueFunction, RobustContinuation,
       RobustPlanningError, RobustActionBackup, RobustDecision,
       fit_value_function!, backup_action, robust_decision, prepare_decision!,
       RobustPolicy, RobustPolicyUpdater, reset_robust_policy!,
       RecedingHorizonRobustPolicy, RecedingHorizonRobustUpdater, configuration_stratified_particles

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
include("adaptive/hyperstates.jl")

"""
    DroneSurveillancePOMDP(; size=(5, 5), region_A=[1, 1],
        region_B=[size[1], size[2]], initial_heading=0,
        obstacles=random_obstacles(size; exclude=[region_A, region_B]),
        camera=KnownCamera(), discount_factor=0.95)

Adaptive robust drone model with hyperstate (configuration, fault mode, counts).
The camera reports noisy classes for three cells ahead. Commands are relative to
heading; all six commands are available at every active pose. Goal entry earns +100, boundary/obstacle collision
-100, and other active transitions -1. Goal/collision terminate the episode.
Learned transitions require an explicit selected row; the physical simulator
executes hidden permanent action faults separately. There is no finite global
state enumeration because count histories are unbounded.
"""
@with_kw mutable struct DroneSurveillancePOMDP <: POMDP{AdaptiveState, Int64, DSObservation}
    size::Tuple{Int64, Int64} = (5, 5)
    region_A::DSPos = [1, 1]
    region_B::DSPos = [size[1], size[2]]
    initial_heading::Int64 = 0
    obstacles::Vector{DSPos} = random_obstacles(size; exclude=[region_A, region_B])
    @assert length(unique(obstacles)) == length(obstacles) "obstacle cells must be distinct"
    @assert all(pos -> 1 <= pos[1] <= size[1] && 1 <= pos[2] <= size[2], obstacles) "obstacles must be inside the grid"
    @assert region_A ∉ obstacles && region_B ∉ obstacles "start and goal must be free"
    camera::KnownCamera = KnownCamera()
    discount_factor::Float64 = 0.95
end

include("adaptive/state_construction.jl")

configuration_terminal(pomdp, r::DSState) = r == UNSAFE_CONFIG || r.quad == pomdp.region_B
POMDPs.isterminal(pomdp::DroneSurveillancePOMDP, s::AdaptiveState) = is_terminal(s)
POMDPs.discount(pomdp::DroneSurveillancePOMDP) = pomdp.discount_factor
include("adaptive/actions.jl")
include("adaptive/faults.jl")
include("adaptive/successors.jl")

# Boundary-risk diagnostic only; this never restricts action availability.
function command_in_bounds(pomdp::DroneSurveillancePOMDP, config::DSState, a::Integer)
    1 <= a <= N_ACTIONS || return false
    x, y = commanded_config(config, a).quad
    return 1 <= x <= pomdp.size[1] && 1 <= y <= pomdp.size[2]
end

"""
    actions(pomdp, s::Union{DSState,AdaptiveState})

Return all six commands at every active configuration. Outward and obstacle-directed
commands are evaluated through their transition/reward consequences: an executed
collision enters UNSAFE_CONFIG and earns -100; staying costs -1.
No pose-dependent action mask or hidden fault mode restricts the action set.
Terminal states have no available actions.
"""
function POMDPs.actions(pomdp::DroneSurveillancePOMDP, s::DSState)
    configuration_terminal(pomdp, s) && return Int64[]
    x, y = s.quad
    1 <= x <= pomdp.size[1] && 1 <= y <= pomdp.size[2] ||
        throw(ArgumentError("source configuration must be inside the grid"))
    return collect(actions(pomdp))
end

function POMDPs.actions(pomdp::DroneSurveillancePOMDP, s::AdaptiveState)
    source = adaptive_state(pomdp, s.config, s.mode)
    s.config == source.config && s.outcome == source.outcome ||
        throw(ArgumentError("source configuration or outcome is inconsistent with the map"))
    is_terminal(source) && return Int64[]
    return actions(pomdp, s.config)
end

include("adaptive/hdr.jl")
include("adaptive/locomotion.jl")
include("adaptive/row_minimization.jl")
include("adaptive/parameter_update.jl")
include("adaptive/transition.jl")
include("adaptive/reward.jl")
include("adaptive/observation.jl")

"""Reuse the camera for active hyperstates; goal/collision states emit the terminal observation."""
function POMDPs.observation(pomdp::DroneSurveillancePOMDP, a::Int64, sp::AdaptiveState)
    isterminal(pomdp, sp) && return Deterministic(TERMINAL_OBSERVATION)
    return observation(pomdp, a, sp.config)
end

include("adaptive/belief.jl")
include("adaptive/planning.jl")
include("adaptive/mixture_master.jl")
include("adaptive/robust_backup.jl")
include("adaptive/particles.jl")
include("adaptive/robust_policy.jl")

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

include("adaptive/episode_logging.jl")
include("adaptive/timing.jl")
include("adaptive/simulation.jl")

function POMDPTools.render(pomdp::DroneSurveillancePOMDP, step)
    nx, ny = pomdp.size
    step_number = get(step, :t, nothing)
    frame_phase = get(step, :phase, nothing)
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
    quad = s === nothing || s == UNSAFE_CONFIG ? nothing :
        render_quad(cell_ctx(s.quad, pomdp.size), s.heading)
    # Reserve a header above the map so the step counter never obscures a cell.
    header_height = step_number === nothing ? 0h : 0.08h
    sz = min(w, h-header_height)
    scene = compose(context((w - sz)/2, header_height+(h-header_height-sz)/2, sz, sz), quad, grid, outline)
    label_text = frame_phase == :initial ? "Initial state (no actions)" : "Step $step_number"
    label = step_number === nothing ? nothing : compose(context(),
        Compose.text(0.5,0.04,label_text,hcenter,vcenter),
        fontsize(14pt),font("DejaVu Sans"),fill("black"))
    background = compose(context(), rectangle(), fill("white"), stroke(nothing))
    return compose(context(), label, scene, background)
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
