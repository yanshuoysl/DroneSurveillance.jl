# Run from the repository root:
# julia --project=test test/test_visualization_adaptive.jl [seed] [H] [output.gif]
# The standalone module also needs the main project's environment dependencies.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using Random
using POMDPs
using POMDPTools
using POMDPGifs
import Cairo, Fontconfig # Enable Compose's PNG backend for GIF frames.

include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
using .DroneSurveillanceAdaptive
include("adaptive_demo_route.jl")

# A fresh map on every run; pass an integer argument to reproduce a layout.
rng = isempty(ARGS) ? Random.default_rng() : MersenneTwister(parse(Int, ARGS[1]))
H = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
waypoint = DSPos(3, 3)
pomdp, route = random_demo_environment(rng, waypoint)
s0 = initial_adaptive_state(pomdp)

# Visit (3, 3) first, then continue east and north to region B.
# Plan a demonstration route around the known map, not from noisy camera readings.
policy = PlaybackPolicy(route)

# Explicit reliable-motion model for this route demonstration: assign all mass
# to commanded motion with unchanged mode. Both motion outcomes and every
# admissible next mode remain in the support. This chooses an HDR-admissible
# candidate, not a worst-case policy or a posterior-mean transition model.
function demo_transition(pomdp, s, a)
    hdr = locomotion_mode_uncertainty(pomdp, s, a; epsilon=0.1, rng=rng)
    commanded_index = findfirst(pair -> pair[1] != s.config && pair[2] == s.mode,
                                hdr.successors)
    theta = zeros(length(hdr.successors))
    theta[commanded_index] = 1.0
    return robust_transition(pomdp, s, a, theta; hdr=hdr)
end

# The policy is scripted; no hidden mode/counts are supplied to it and no belief
# learning is claimed. A future AR planner can supply its own policy/updater.
result = simulate_adaptive(pomdp, policy, NothingUpdater(), nothing;
                           H=H, transition_kernel=demo_transition,
                           initial_state=s0, rng=rng)
hist = result.history
# Check the recorded route visits the waypoint before arriving at the goal.
positions = [step.sp.config.quad for step in eachstep(hist, "(t,sp)")]
waypoint_step = findfirst(==(waypoint), positions)
goal_step = findfirst(==(pomdp.region_B), positions)
if result.reason == :goal
    @assert waypoint_step !== nothing && goal_step !== nothing && waypoint_step < goal_step
    @assert goal_step == length(hist) # No post-goal rotations or extra updates.
    @assert positions[end] == pomdp.region_B
end
@assert length(hist) <= H
@assert sum(values(result.final_state.counts.increments); init=0) == length(hist)
@assert all(pos -> pos ∉ pomdp.obstacles, positions)
println("Obstacle cells: ", Tuple.(pomdp.obstacles))
println("Stopped: ", result.reason, "; steps=", result.steps,
        "; total reward=", sum(step.r for step in hist; init=0.0))
filename = length(ARGS) >= 3 ? abspath(ARGS[3]) :
    normpath(joinpath(@__DIR__, "..", "drone_adaptive.gif"))
for step in eachstep(hist, "(t,a,sp,o,r)")
    println("Step ", step.t, ": action=", step.a,
            ", position=", Tuple(step.sp.config.quad),
            ", camera (left, center, right)=", observation_labels(step.o),
            ", reward=", step.r)
end
if isempty(hist)
    println("No actions executed; GIF skipped.")
else
    makegif(pomdp, hist; filename, spec="(sp,o)", fps=2, show_progress=false)
    println("Saved visualization to ", filename)
end
