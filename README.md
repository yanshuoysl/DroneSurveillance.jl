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
The +1 reward for occupying region B is retained. Commands that would leave
the grid are invalid. Reaching region B does not terminate this legacy DSState
simulator; the augmented model below treats goal entry as terminal.

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
at `(5, 5)`, where the episode ends immediately. Two obstacles are randomized each run while
keeping `(3, 3)`, the start, and goal free. Disconnected layouts are resampled.
The demonstration route uses the known map to avoid obstacles, rather than
reacting to noisy camera readings. Ordinary cells in the FOV are blue, survey
regions remain green, and obstacles are orange. The script runs on AdaptiveState,
using an explicit HDR-checked reliable-motion row and recording count updates
and transition rewards. It does not perform worst-case row selection or belief
learning. It prints the termination reason, executed steps, rewards, obstacle
cells, positions, and sampled camera readings. Pass a seed, optional step limit H
(default 100), and optional GIF path:

```sh
julia --project=test test/test_visualization_adaptive.jl 42
julia --project=test test/test_visualization_adaptive.jl 42 3 /tmp/drone_horizon.gif
```

### Adaptive successor support and Dirichlet rows

`feasible_successors(pomdp, r, m, a)` returns the ordered feasible `(r′, m′)`
pairs for each of the six actions. For an active source, support is the Cartesian
product of `{stay, commanded configuration}` and `successor_modes(m)`: both
configurations are included for every admissible next mode, regardless of that
action's fault flag. Modes stay unchanged or gain one fault, with no recovery.
Thus nominal mode has 14 pairs and a mode with one existing fault has 12.
The joint probabilities remain unknown; Cartesian support does not impose
probabilistic independence. No sideways slip or unintended rotation is included.

Outward commands are excluded by `actions(pomdp, s)` and rejected by the
transition and map-aware count APIs, including their potential stay branch.
They have no feasible successor row. Availability depends on configuration,
without consulting fault flags or counts. Rotations remain available at active
boundary cells. `actions(pomdp)` still lists all six IDs for indexing.
For a `DiscreteBelief`, `actions(pomdp, b)` returns the intersection of valid
commands across configurations with positive probability. Use a policy that
consults these state/belief-dependent actions; for example,
`RandomPolicy(pomdp; updater=DiscreteUpdater(pomdp))` for the DSState simulator.
The default `RandomPolicy` with `NothingUpdater` samples global action IDs and
does not enforce boundary availability.

The support uses the augmented state's terminal conventions: obstacle contact
maps to `UNSAFE_CONFIG`, and goal/unsafe states are absorbing.
The existing `DSState` simulator still treats obstacle contact as blocked motion.
The support specifies possible outcomes. The augmented transition kernels below
assign probabilities without changing the existing deterministic DSState model.

```julia
include("src/DroneSurveillanceAdaptive.jl")
using .DroneSurveillanceAdaptive

pomdp = DroneSurveillancePOMDP(obstacles=DSPos[])
r = DSState((3, 3), 0) # north-facing
m = NOMINAL_MODE
a = ACTIONS_DICT[:move_right]

successors = feasible_successors(pomdp, r, m, a)
@assert is_feasible_successor(pomdp, r, m, a, DSState((4, 3), 0), m)
@assert !is_feasible_successor(pomdp, r, m, a, DSState((2, 3), 0), m)

phi = DirichletCounts()
key = CountKey(pomdp, r, m, a, DSState((4, 3), 0), m)
learned = increment_count(pomdp, phi, key) # new snapshot; phi is unchanged
row = count_row(pomdp, learned, r, m, a)
hdr = build_row_hdr(pomdp, adaptive_state(pomdp, r, m, learned), a; epsilon=0.05)
```

Impossible transitions are structural zeros, excluded from the Dirichlet row.
The map-aware `CountKey`, `count_value`, and `increment_count` overloads reject
them with `ArgumentError`; `count_row(pomdp, ...)` requires the complete feasible
support. Use these overloads for model counts. The original storage APIs without
`pomdp`, and the HDR overload with an explicit successor list, retain their
looser checks for illustrative fixtures. The new HDR overload above computes
the validated support automatically.

Run the support tests with `julia --project=. test/test_successors.jl`.

### Locomotion and mode dynamics

The first AR-POMDP transition factor is a family of joint configuration/mode
kernels indexed by an admissible row, not a fixed posterior-mean transition.
For an active source, its flow is:

```text
(r,m,phi,a) -> S(r,m,a) -> row counts alpha -> U(phi)
           -> supplied theta in U(phi) -> distribution over (r',m')
```

`locomotion_mode_uncertainty(pomdp, s, a; epsilon=...)` returns a `DirichletHDR`
representing the continuous uncertainty set U(phi). It uses the full feasible
support and the source hyperstate's counts. Monte Carlo samples calibrate the
HDR threshold only; they do not select a physical transition row. Coverage
retains the existing Monte Carlo approximation.

`locomotion_mode_transition(pomdp, s, a, theta; hdr=U)` validates an admissible
row and returns a distribution over **only `(DSState, FaultMode)` pairs**.
Theta follows `U.successors` order. Every active source requires an HDR matching
its complete support and current row counts. The function does not create phi',
increment any count, or default to the posterior mean. Selecting a worst-case
row belongs in the robust planner. Cartesian support does not impose independence
between configuration and mode probabilities.

```julia
s = adaptive_state(pomdp, DSState((3, 3), 0), NOMINAL_MODE)
a = ACTIONS_DICT[:move_right]
U = locomotion_mode_uncertainty(pomdp, s, a; epsilon=0.05,
                               rng=MersenneTwister(42))

# Explicit candidate for demonstration under the all-one prior's full simplex.
# This is not an estimate of the true dynamics or a worst-case optimization.
theta = zeros(length(U.successors))
theta[1] = 0.25 # stay, nominal next mode
theta[8] = 0.75 # move right, nominal next mode
physical = locomotion_mode_transition(pomdp, s, a, theta; hdr=U)
rp, mp = rand(MersenneTwister(43), physical)
@assert isempty(s.counts.increments) # sampling this factor never updates phi
```

Goal/unsafe sources have no learned uncertainty row. Use
`locomotion_mode_transition(pomdp, s, a, [1])` without an HDR for their absorbing
configuration/mode distribution. Run the focused tests with
`julia --project=. test/test_locomotion.jl`.

### Parameter-update transition

`parameter_update_transition(pomdp, s, a, rp, mp)` implements the second factor
of the AR-POMDP transition, conditioned on a feasible latent `(r,m,a,rp,mp)`.
It returns a deterministic distribution over the full next count vector:
probability 1 at `phi + delta(r,m,a,rp,mp)`, and 0 at every other count vector.

```julia
using Distributions: pdf

# rp, mp come from a latent locomotion/mode branch, as in the example above.
parameters = parameter_update_transition(pomdp, s, a, rp, mp)
phi_prime = rand(parameters) # deterministic: always the same updated snapshot
@assert pdf(parameters, phi_prime) == 1.0
@assert pdf(parameters, s.counts) == 0.0 # this example starts in an active state
sp = adaptive_state(pomdp, rp, mp, phi_prime)
```

Only the count indexed by `(r,m,a,rp,mp)` increases by one, including when the
robot stays in place or enters a goal/unsafe state. All other action rows,
destinations, modes, and source rows remain unchanged. The original count vector
is not mutated, and different hypotheses own separate snapshots. The explicit
argument overload is
`parameter_update_transition(pomdp, r, m, phi, a, rp, mp)`.

This factor uses no theta, uncertainty set, normalization, or observation. It is
deterministic conditioned on a latent transition, which does not mean the robot
knows which count history occurred. Observation-based inference over competing
histories belongs in the belief updater. Infeasible conditioning pairs are
rejected. For an already terminal source, only its absorbing destination is
allowed and its count vector stays unchanged. Run the focused tests with
`julia --project=. test/test_parameter_update.jl`.

### Combined transition kernels

The combined kernels use the separate parameter-update factor for each latent
locomotion/mode successor. `adaptive_transition(pomdp, s, a)`
is a Bayesian baseline, separate from the AR-POMDP's uncertain first factor.
It returns a distribution over next
`AdaptiveState`s using the Dirichlet posterior predictive row. If
`s = (r,m,phi)`, its transition probability for a feasible `(r′,m′)` is
`phi[key] / sum(phi_row)`, and the next state contains `phi + delta(key)`.
Every joint pair in the Cartesian support has its own probability; configuration
and mode probabilities are not factorized. With all-one counts, the nominal
14-entry row initially assigns probability `1/14` to every joint pair.

```julia
using Random

s = adaptive_state(pomdp, r, m, learned)
predictive = posterior_predictive_row(pomdp, s, a)
@assert length(predictive.successors) == 14
next_distribution = adaptive_transition(pomdp, s, a)
sp = rand(MersenneTwister(42), next_distribution)
# s.counts is unchanged; sp has exactly one additional transition count.
```

An explicit row can be supplied as `adaptive_transition(pomdp, s, a, theta)`.
It must be a probability vector in `feasible_successors` order. This supports
fixed physical-model rows or posterior-sampled rows, and checks the simplex.
The existing `robust_transition(pomdp, s, a, theta; hdr=hdr)` now delegates its
first factor to `locomotion_mode_transition`, then applies
`parameter_update_transition` to construct each full next hyperstate. The
resulting probability is the physical pair probability multiplied by the
deterministic parameter factor, exactly as in the AR-POMDP factorization.

Every branch owns a separate count snapshot, including stay and goal/unsafe
entry branches. Goal/unsafe source states return an absorbing distribution with
unchanged mode and counts; they have no learned or HDR row. The kernels validate
source outcome labels against the map and reject malformed probability vectors.

These are conditional kernels over hidden-state hypotheses. They do not update
the robot's belief using true simulator states. The observation-based belief
updater, robust planner, and augmented-state POMDP wrapper remain separate work;
`POMDPs.transition` on the existing DSState environment retains its original
behavior. Run kernel tests with
`julia --project=. test/test_adaptive_transition.jl`.

### AR-POMDP transition reward

`adaptive_reward(pomdp, s, a, sp)` implements R(s,a,s') for augmented states:

- +100 when an active source enters the goal;
- -100 when an active source collides with an obstacle;
- -1 otherwise, including ordinary movement, rotation, and staying in place;
- 0 on an already terminal state's absorbing transition.

The goal/collision rewards are the full transition rewards; no additional -1
is added to them. A goal/collision reward is therefore paid only on entry. The
numeric reward is independent of the Dirichlet counts and candidate transition
row. It does not mutate either state's counts. The equivalent standard interface
is `reward(pomdp, s::AdaptiveState, a, sp::AdaptiveState)`.

```julia
using POMDPs: reward

s = adaptive_state(pomdp, DSState((4, 5), 0))
a = ACTIONS_DICT[:move_right]
rp, mp = DSState(pomdp.region_B, 0), NOMINAL_MODE
parameters = parameter_update_transition(pomdp, s, a, rp, mp)
sp = adaptive_state(pomdp, rp, mp, rand(parameters))
@assert adaptive_reward(pomdp, s, a, sp) == 100.0
@assert reward(pomdp, s, a, sp) == 100.0
@assert reward(pomdp, sp, a, sp) == 0.0
```

Feasible unsafe entries are obstacle collisions and receive -100. Outward
boundary commands are invalid, so they have no reward transition. A stay before
an obstacle receives -1,
since no collision occurred. Inputs must be canonical and form a feasible
configuration/mode transition. The legacy three-argument DSState reward remains
separate. Run reward tests with `julia --project=. test/test_adaptive_reward.jl`.

### Augmented simulation termination

`POMDPs.isterminal(pomdp, s::AdaptiveState)` recognizes goal and collision
outcomes. `termination_reason(pomdp, s, steps, H)` returns `:goal`, `:collision`,
`:horizon`, or `nothing`. Here `steps` counts completed actions, including
staying and rotating. Goal/collision take precedence on step H. The horizon is
external to the hyperstate and count keys; it does not create a physical
terminal state or an extra reward.

The visualization example uses the dedicated augmented runner:

```julia
result = simulate_adaptive(pomdp, policy, NothingUpdater(), nothing;
    H=H, transition_kernel=demo_transition, initial_state=s0, rng=rng)
hist = result.history
println(result.reason, ": ", result.steps, " completed actions")
```

`demo_transition` is defined in `test/test_visualization_adaptive.jl`. The
runner requires an explicit kernel returning a distribution over full next
hyperstates, including phi'. It records the sampled transition, reward, camera
observation, and belief update before checking whether another action may run.
The terminal entry count/reward is retained, with no action H+1 or repeated
terminal reward. H=0 and initially terminal states return an empty history.

The result contains `history` (a POMDPTools.SimHistory), `final_state`,
`final_belief`, `reason`, `steps`, `terminated` (goal/collision), and `truncated`
(horizon). Active hyperstates reuse the camera model; terminal ones emit the
terminal observation. Rendering supports augmented states and shows the drone
at its final goal position.

The policy receives only the supplied belief. The updater receives `(b,a,o)`;
the true mode/count history is not passed to either. The example is a scripted
route with NothingUpdater. A Bayesian belief updater and robust planner are
still separate work, as is a POMDP wrapper with AdaptiveState as its declared
state type for general POMDPTools simulators. Run termination tests with
`julia --project=. test/test_adaptive_termination.jl`.

### Hidden episode fault schedule

`sample_fault_schedule(H, N_f; rng)` samples K uniformly from the integers
`0:min(N_f,H,6)`, then chooses K distinct fault steps uniformly from `1:H` and
returns them in chronological order. For H=100 and N_f=2, zero, one, and two
scheduled faults each have probability 1/3. Conditional on K, every K-element
set of steps is equally likely. This does not give all schedules of different
sizes equal probability. Zero horizon or zero fault budget gives no events.

```julia
fault_steps = sample_fault_schedule(100, 2; rng=MersenneTwister(42))
K = length(fault_steps)
```

This implements the hidden schedule sampler only. Fault activation and the
uniform choice of a healthy action at each event still need connecting to the
simulation. A scheduled event will take effect before its step's command is
executed, and events after early termination will never occur. The sampler
does not change the learner's Cartesian successor support or remove frozen
actions from the robot's action set. Run its tests with
`julia --project=. test/test_fault_schedule.jl`.

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
