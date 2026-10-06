# Run from the repository root:
# julia --project=test test/test_visualization_adaptive.jl
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
waypoint = DSPos(3, 3)
pomdp, route = random_demo_environment(rng, waypoint)
s0 = rand(rng, initialstate(pomdp))

# Visit (3, 3) first, then continue east and north to region B.
# Rotate clockwise in place at the destination to show all four headings.
# Plan a demonstration route around the known map, not from noisy camera readings.
append!(route, fill(ACTIONS_DICT[:rotate_clockwise], 6))
policy = PlaybackPolicy(route)
o0 = rand(rng, observation(pomdp, route[1], s0))

hist = simulate(HistoryRecorder(max_steps=length(route), rng=rng), pomdp, policy,
                PreviousObservationUpdater(), o0, s0)
# Check the recorded route visits the waypoint before arriving at the goal.
positions = [step.sp.quad for step in eachstep(hist, "(t,sp)")]
waypoint_step = findfirst(==(waypoint), positions)
goal_step = findfirst(==(pomdp.region_B), positions)
@assert waypoint_step !== nothing && goal_step !== nothing && waypoint_step < goal_step
@assert positions[end] == pomdp.region_B
@assert all(pos -> pos ∉ pomdp.obstacles, positions)
println("Obstacle cells: ", Tuple.(pomdp.obstacles))
filename = normpath(joinpath(@__DIR__, "..", "drone_adaptive.gif"))
makegif(pomdp, hist; filename, spec="(sp,o)", fps=2, show_progress=false)
for step in eachstep(hist, "(t,a,sp,o)")
    println("Step ", step.t, ": action=", step.a,
            ", position=", Tuple(step.sp.quad),
            ", camera (left, center, right)=", observation_labels(step.o))
end
println("Saved visualization to ", filename)
