# DroneSurveillance.jl

[![Build status](https://github.com/JuliaPOMDP/DroneSurveillance.jl/workflows/CI/badge.svg)](https://github.com/JuliaPOMDP/DroneSurveillance.jl/actions)
[![codecov](https://codecov.io/gh/juliapomdp/DroneSurveillance.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliapomdp/DroneSurveillance.jl)


Implementation of a drone surveillance problem<sup>1</sup> with the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl).

<sup>1</sup> M. Svoreňová, M. Chmelík, K. Leahy, H. F. Eniser, K. Chatterjee, I. Černá, C. Belta, "
Temporal logic motion planning using POMDPs with parity objectives: case study paper", *International Conference on Hybrid Systems: Computation and Control (HSCC)*, 2015.

![drone_surveillance.gif](drone_surveillance.gif)

## Installation

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/DroneSurveillance.jl"))
```


## Example

To run the visualization script from a local checkout, first set up its test
environment (run these commands from the repository root):

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test test/test_visualization.jl
```

The script writes `test.gif` in the current directory, then solves the model with
SARSOP, which writes `model.pomdpx` and `policy.out` there. GIF rendering requires
`import Cairo, Fontconfig` to enable Compose's PNG backend.

```julia
using DroneSurveillance
using POMDPs

# import a solver from POMDPs.jl e.g. SARSOP
using SARSOP

# for visualization
using POMDPGifs
import Cairo, Fontconfig

pomdp = DroneSurveillancePOMDP() # initialize the problem 

solver = SARSOPSolver(precision=1e-3) # configure the solver

policy = solve(solver, pomdp) # solve the problem

makegif(pomdp, policy, filename="out.gif")
```


## Problem Description

### DroneSurveillanceAdaptive

`src/DroneSurveillanceAdaptive.jl` is a separate copy of the environment for
experiments without a ground agent. Load it from the repository root:

```julia
include("src/DroneSurveillanceAdaptive.jl")
using .DroneSurveillanceAdaptive
using POMDPs, POMDPTools, Random

pomdp = DroneSurveillanceAdaptive.DroneSurveillancePOMDP(size=(5, 5))
s = rand(initialstate(pomdp)) # DSState contains `quad` and `heading`
sp = rand(transition(pomdp, s, ACTIONS_DICT[:forward])) # north when heading is 0
o = rand(observation(pomdp, ACTIONS_DICT[:forward], sp))
@show observation_labels(o) # front-left, front-center, front-right
render(pomdp, (s=sp,))
```

The variant has four headings per free drone position plus a terminal state (93
states on a 5×5 grid with two obstacles). Headings are clockwise degrees from north: 0 (north), 90 (east),
180 (south), and 270 (west). Set `initial_heading` to choose the starting heading;
the default is 0. `DSState([x, y], heading)` constructs a state, and omitting the
heading defaults to north. Motion and the initial state are deterministic.
The camera returns noisy cell classes rather than the true position or heading.
Ground-agent movement, collision penalties, and ground-agent rendering are removed.
The +1 reward for occupying region B is retained; leaving
the grid terminates the episode. Reaching region B does not terminate it.

By default, each new environment randomly places two distinct obstacles, excluding
the start and goal. Obstacle positions belong to the static map in
`pomdp.obstacles`; the drone state still contains only its position and heading.
Moving into an obstacle leaves the drone in place. Obstacles are shown in orange
and excluded from the traversable state space. Set `obstacles=DSPos[]` to
use a map without obstacles, or supply fixed obstacle cells.

```julia
obstacles = random_obstacles((5, 5); exclude=[DSPos(1, 1), DSPos(5, 5)],
                             rng=MersenneTwister(42))
pomdp = DroneSurveillancePOMDP(obstacles=obstacles)
```

State enumeration, indexing, and initialization live in `src/adaptive/states.jl`.
Action definitions and indexing live in `src/adaptive/actions.jl`. Both files are
included by `src/DroneSurveillanceAdaptive.jl`, where the `DSState` type is defined.
Random obstacle generation lives in `src/adaptive/obstacles.jl`.

Camera types and the known sensor model live in `src/adaptive/camera.jl`.
FOV geometry, observation enumeration, and observation probabilities live in
`src/adaptive/observation.jl`.

The six actions are:

| ID | Name in `ACTIONS_DICT` | Effect |
| --- | --- | --- |
| 1 | `:forward` | Move one cell in the direction the drone faces |
| 2 | `:backward` | Move one cell opposite the direction it faces |
| 3 | `:move_left` | Move one cell to the drone's left |
| 4 | `:move_right` | Move one cell to the drone's right |
| 5 | `:rotate_clockwise` | Turn 90° clockwise in place |
| 6 | `:rotate_counterclockwise` | Turn 90° counterclockwise in place |

Translations retain the heading. Rotations retain the position. There is no hover
action. The yellow triangle on the drone shows its heading.

The FOV is exactly three cells in a row immediately ahead of the drone, ordered
front-left, front-center, front-right. It rotates with the heading and excludes
the cell under the drone and cells behind it. For example, from `(3, 3)` facing
north, the visible cells are `(2, 4)`, `(3, 4)`, and `(4, 4)`.

`DSObservation` contains three class IDs: 1 = ordinary cell, 2 = survey region
(A or B), 3 = outside the grid, 4 = obstacle. The terminal observation is `[0, 0, 0]`.
There are 64 possible camera triples plus this terminal observation. Cells outside the
grid remain in the three-slot observation rather than being dropped.

The default `KnownCamera()` reports the correct class with probability 0.90 and
each of the three incorrect classes with probability `0.10/3`. The three readings are conditionally
independent, so the likelihood of a triple is the product of its per-cell
probabilities. These are configurable simulation assumptions, not measured
camera performance. The model is known to the planner; individual readings are
random. Configure uniform accuracy or a custom confusion matrix with:

```julia
pomdp = DroneSurveillancePOMDP(camera=KnownCamera(correct_probability=0.85))

# Rows: true class; columns: reported class.
# Order: ordinary, survey, outside, obstacle. Each row must sum to 1.
camera = KnownCamera([0.90 0.04 0.02 0.04;
                      0.10 0.80 0.05 0.05;
                      0.02 0.01 0.95 0.02;
                      0.10 0.03 0.02 0.85])
pomdp = DroneSurveillancePOMDP(camera=camera)
```

Position and heading still belong to the environment state, but are not directly
reported by the camera. With the current known initial pose and deterministic
motion, a policy can track its pose from action history. The grid renderer shows
the true map for debugging; its highlighted cells show the camera's FOV.

Run its tests with `julia --project=. test/test_adaptive.jl`.

To visualize the variant, run
`julia --project=test test/test_visualization_adaptive.jl` after setting up the
test environment as described above. This creates `drone_adaptive.gif` in the
repository root, showing the drone visiting `(3, 3)` before continuing to region B
at `(5, 5)`, then rotating in place. Two obstacles are randomized each run while
keeping `(3, 3)`, the start, and goal free. Disconnected layouts are resampled.
The demonstration route uses the known map to avoid obstacles, rather than
reacting to noisy camera readings. Ordinary cells in the FOV are blue, survey
regions remain green, and obstacles are orange. The script prints obstacle cells,
positions, and sampled camera readings. Pass an integer seed to reproduce a layout:

```sh
julia --project=test test/test_visualization_adaptive.jl 42
```

### Original environment

A drone must survey two region (in green) while avoiding to fly over a ground agent. The drone has a limited field of view. 

- **States**: position of the drone and the agent in the grid world
  
- **Actions**: Moving up, down, left, right, or hovering over the current cell.
  
- **Transition model**: The drone moves deterministically to the desired cell depending on the action chosen. The agent follows a random policy, it can stay in the same cell or move to a neighboring cell with equal probability. In addition the agent cannot enter the area to survey.
  
- **Observation model**: if the agent is in the field of view of the drone, the drone observes the position of the agent within its field of view with probability 1, otherwise it does not observe the agent (value `OUT`).
  
- **Initial state**: the drone starts in the bottom left corner, the agent is initialized outside the field of view of the drone.
  
- **Reward model**: a naive reward model is implemented, the drone receives +1 for reaching the second area to survey, and -1 if it flies over the ground agent.


### `DroneSurveillancePOMDP` Parameters

- constructor: `DroneSurveillancePOMDP(kwargs...)`
- keyword arguments:
  - `size::Tuple{Int64, Int64} = (5,5)` size of the grid world
  - `region_A::DSPos = [1, 1]` first region to survey, initial state of the quad
  - `region_B::DSPos = [size[1], size[2]]` second region to survey
  - `fov::Tuple{Int64, Int64} = (3, 3)` size of the field of view of the drone
  - `agent_policy::Symbol = :random` policy of the other agent, only random is implemented
  - `terminal_state::DSState = DSState([-1, -1], [-1, -1])` a sentinel state to encode terminal states
  - `discount_factor::Float64 = 0.95` the discount factor

### Internal types:

- `DSPos` represents a position in the grid as a static array of 2 integers
- `DSState` represents the state of the environment, the field `quad` represents the position of the drone and the file `agent` the position of the ground agent
