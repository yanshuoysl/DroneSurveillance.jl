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

pomdp = DroneSurveillanceAdaptive.DroneSurveillancePOMDP(obstacles=DSPos[])
s = rand(initialstate(pomdp)) # AdaptiveState(config, mode, counts, outcome)
env = HiddenFaultEnvironment(1, 0; fault_steps=[])
sp = physical_step(pomdp, env, s, ACTIONS_DICT[:forward], 1).sp
o = rand(observation(pomdp, ACTIONS_DICT[:forward], sp))
@show observation_labels(o) # front-left, front-center, front-right
render(pomdp, (s=sp,))
```

The AR model declares `AdaptiveState` as its POMDP state type. Its configuration
component `DSState([x,y], heading)` contains position and clockwise heading:
0 north, 90 east, 180 south, 270 west. The initial hyperstate has a known start
configuration, nominal mode, and implicit count two at each feasible coordinate.
`src/adaptive/hyperstates.jl` defines these types; map-aware constructors and
initialization live in `src/adaptive/state_construction.jl`. There is no finite
`states`/`stateindex` interface for the unbounded count space. The old configuration
state enumeration and deterministic `POMDPs.transition` have been removed from
the AR module. Select an explicit learned row or use the physical fault runner;
there is no implicit model attached to `POMDPs.transition`.

Goal entry earns +100, boundary/obstacle collision earns -100, and other active
steps earn -1. Goal and collision terminate the episode. All six actions are
available at every nonterminal pose and belief; boundary risks affect their values.
The separate original `DroneSurveillance` package is unchanged.

By default the known map has two randomly placed obstacles, excluding start and
goal. `obstacles=DSPos[]` disables them. Obstacles render orange. Actual entry
into one causes collision; a frozen action that stays before it does not collide.

```julia
obstacles = random_obstacles((5, 5); exclude=[DSPos(1, 1), DSPos(5, 5)],
                             rng=MersenneTwister(42))
pomdp = DroneSurveillancePOMDP(obstacles=obstacles)
```

Action definitions live in `src/adaptive/actions.jl`, and map sampling in
`src/adaptive/obstacles.jl`.

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

Position, heading, mode, and count history are not directly reported by the
camera. The renderer shows physical truth for evaluation only.

After reviewing `src/DroneSurveillanceAdaptive.jl`, the complete robust example is:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/run_adaptive.jl 42 2
```

Arguments are seed, episode H, output root, and N_f, plus optional
`--lookahead L` (default 2), `--particles N`, `--fault_time_step T`, `--fault_type A`, `warmup=true|false`, and
`tied_action_rdn=true|false` (both booleans default to false).
The equivalent `--warmup=true` / `--warmup false` and
`--tied_action_rdn=true` / `--tied_action_rdn true` forms are also accepted.
Boolean values are case-insensitive (`True` and `False` work too).
Each invocation creates a unique
`run_seed42_H2_L2_<suffix>` folder inside `test/output`, independent of the current
working directory. An explicit third argument changes the parent output root;
the new run folder is still created inside it. The chosen folder is printed
before fitting and returned as `result.output_directory`. The default H=2 example starts
at the bottom-left cell (1,1), with goal (5,5) and a fixed obstacle at (3,5).
H=2 is a short execution check: the goal requires at least eight successful moves.
H controls physical execution and fault scheduling. L controls each local plan:
`min(L, H - completed_steps)`. Before every new action, the explicit
`RecedingHorizonRobustPolicy` fits a local finite-continuation approximation from
the current exact belief, selects robust transition rows, executes
hidden physical faults, and performs the exact model-conditioned belief update.
There is no forced route or waypoint. Repeated runs get separate folders.

To fix when a random action develops a fault, use:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 2 --particles 128 --fault_time_step 4
```

This schedules exactly one fault at step 4, with no additional random fault
times. A healthy action is chosen uniformly at that step and its fault is
activated before executing command 4. The action choice remains reproducible
from the run seed. `--fault_time_step=4` is also accepted. The step must be
inside `1:H`, and the fault budget `N_f` must be at least 1. If the episode
ends before that step, the scheduled fault never activates. Omitting the option
preserves the existing random schedule (including its random event count).
Warm-up uses the same requested schedule in a separate episode. The requested
step is returned as `result.fault_time_step` (`nothing` when omitted), and the
actual schedule remains available in `result.fault_steps` and the log's start
event. The schedule remains hidden from the planner.

To select which action is disabled, add `--fault_type`:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 2 --particles 128 --fault_time_step 4 --fault_type 1
```

This permanently disables Forward starting at step 4, including command 4
itself. A disabled command leaves the robot's position and heading unchanged;
other commands continue to work. `--fault_type` requires `--fault_time_step`.
`--fault_type=1` is also accepted. Omitting it retains the random action choice.
The type is returned as `result.fault_type` (`nothing` when omitted), printed
in `console.log`, and recorded as `fault_actions` in the log's start event.
Warm-up uses the same explicit type in its separate episode.

| Fault type | Disabled action |
|---|---|
| 1 | Forward |
| 2 | Backward |
| 3 | Left |
| 4 | Right |
| 5 | Clockwise rotation |
| 6 | Counterclockwise rotation |

To choose randomly among tied best actions, enable the option explicitly:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 2 --particles 128 tied_action_rdn=true
```

With `tied_action_rdn=false`, numerical ties retain the first action in ID order
(Forward if all six tie). With `true`, the policy samples uniformly among
actions tied for the best robust lower bound, using the existing tiny numerical
tolerance. It compares full-precision values, not the rounded chart labels;
a unique best action is always selected. A separate RNG, seeded from the run
seed, advances across decisions without changing the fault-schedule, observation,
HDR, or particle-filter RNG streams. Repeated queries for a pending decision
reuse its selected action and transition rows. Resetting the policy restarts
the tie RNG, and warm-up uses a separate fresh policy. The setting and tie seed
are saved in `episode.jsonl`; the setting also appears in `console.log`.

Programmatically, `run_adaptive_example(...; tied_action_rdn=true)` overrides
the command-line value. Both `RobustPolicy(vf; tied_action_rdn=true)` and
`RecedingHorizonRobustPolicy(vf; H=30, tied_action_rdn=true)` support the option,
with an optional `tie_seed` keyword (default `vf.seed`). Direct
`robust_decision(...; tied_action_rdn=true, rng=MersenneTwister(42))` calls
accept an explicit RNG. The default remains deterministic in all entry points.

The main example generates the GIF automatically. The older visualization
entry point remains a compatibility shortcut to the same implementation:

```sh
julia --project=test test/test_visualization_adaptive.jl 42 2
```

Both entry points save `console.log`, `episode.jsonl`, `rewards.csv`, `rewards.svg`,
`timing.json`, `drone_steps.gif`, and combined per-step JPGs inside the new folder
(the GIF is skipped if no actions execute). These
commands are alternative ways to run one episode. Previous records are preserved:

```text
test/output/
  run_seed42_H2_L2_<suffix>/
    console.log
    episode.jsonl
    rewards.csv
    rewards.svg
    timing.json
    drone_steps.gif
    steps/
      step_0001.jpg
      step_0002.jpg
      step_0003.jpg  # final state after action 2
```

Merely including either script does not run an
episode. To set H and L separately:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 2
```

This requests up to 30 executed actions and replans with at most two actions of
lookahead each time; it does not pre-expand a 30-step training tree. It is an
explicit receding-horizon approximation. The local terminal value remains zero,
so a short L can miss a goal beyond the window. No terminal heuristic has been
added. Exact belief support still grows, and each local fit retains the existing
resource limits; a 30-step run is not guaranteed to fit those limits.
For programmatic headless runs, pass `generate_gif=false` to
`run_adaptive_example`; this disables both animation and JPGs. Set
`generate_jpgs=true` independently for JPGs without animation. The command-line
default generates both. `console.log` is saved even when `io=nothing` suppresses
terminal output; `result.console_log_path` gives its path.
The combined figure shows the drone map on the left, robust Q-values on the
right, and the current six fault flags below. The GIF starts with decision 1's
initial pose; its path is returned as `result.gif_path`. Decision frames display
`Step t` and the pose BEFORE action t, matching decision t in `console.log`.
After N executed actions, an additional frame displays the actual final state,
labelled `Final state — after action N`. Successful episodes show the drone in
the goal cell. Horizon and collision endings show their recorded outcome.
Both live runs and saved-log replay save N+1 frames, numbered 1 through N+1.

`steps/step_0001.jpg` shows the pose before action 1; `step_NNNN.jpg` matches
the corresponding GIF frame, including the extra final image. Paths are
returned as `result.jpg_directory` and `result.jpg_paths`. An empty episode still
saves its initial JPG as `step_0000.jpg`, labelled `Initial state (no actions)`,
without a GIF. Both media formats come from the same lossless rendered frames;
temporary PNGs are removed after encoding. Combined figures are 1800×1000 pixels.

The Q-value panel shows the logged robust Q-value for each of the six actions.
Numeric values are labelled and the selected action is green. Every frame uses
the same axis scale and action order. Negative bars extend down from zero and
positive bars extend up; higher values are better. Values come directly from
the planner's recorded action backups. The final frame retains the last
executed decision's values, explicitly labelled `Last decision N — action values`;
it does not compute a new decision at the final state.

The fault panel shows Forward, Backward, Left, Right, CW, and CCW in action
order. Green `Nominal` indicators mean the flag is clear; red `Fault flagged`
indicators mean it is set. These are the displayed physical state's simulator
truth, for evaluation only. A fault introduced during action t appears in the
next state frame, including a fault introduced by the final action.

The default export produces one GIF and one image folder. The former separate
`action_values.gif`, `action_values/`, and `result.q_*` return fields are no
longer generated by the episode runner or combined replay. Media export is
performed after core computation timing is saved.

For each decision, `console.log` lists every believed position and heading with
its probability, log probability, and available actions; the belief-wide
available actions; and each boundary-directed command's total outward probability
and a witness pose. These are risk diagnostics, not action vetoes. Simulator truth
before and after the command is logged separately with its available actions.
These diagnostics never enter planning or
filtering. Structured equivalents are recorded in `episode.jsonl`.

Warm-up is disabled by default, so normal command-line runs execute one episode.
For steady-state timing benchmarks, `run_adaptive_example(...; warmup=true)`
first replays the same seeded episode as an untimed compilation warm-up. It then constructs a fresh model,
policy, belief, HDR caches and RNGs for the recorded episode; warm-up does not
train the measured policy or add steps to the recorded history. This means the
benchmark executes the workload twice, but records one episode. Warm-up produces
no files or animation.

Each run now saves `timing.json` alongside its other artifacts. Read
`computation_seconds` for the sum of planning (including fitting and HDR/root
solves), physical simulation/observation/reward, and belief-update wall times,
minus Julia-reported residual compilation. Logging/diagnostic serialization,
startup, model setup, warm-up, reward plotting and GIF generation are outside
these timers. Garbage collection and normal runtime overhead within the phases
remain included. This is a compilation-adjusted elapsed-time estimate, not CPU
time or a claim of cycle-exact compiler accounting. Julia >= 1.11 reports the
compilation counters; older versions save `null` for the adjusted computation
total rather than claiming compilation has been excluded.

`core_wall_seconds` is the raw sum before subtracting compilation;
`compilation_seconds` records that subtraction. The three phase totals are raw
wall times. `episode_wall_seconds` additionally includes diagnostics/logging and
loop bookkeeping, but excludes setup and rendering. Per-step phase timings are
saved as `step_timing` events in `episode.jsonl`; the `stop` event contains the
episode summary. The terminal prints the summary before GIF generation.
Warm-up defaults to false and residual compilation is still reported. Set it
explicitly on the command line, anywhere among the arguments:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 2 --particles 128 warmup=false
```

Use `warmup=true` to enable the replay. These named options do not consume the
positional output directory or fault budget. A programmatic call can override
the command-line setting with `run_adaptive_example(...; warmup=false)` or
`warmup=true`. Direct `simulate_robust`
calls record timings but do not perform warm-up automatically. Timings are not
retroactively recoverable for older experiment folders.

If the episode log already exists and you only want its GIF and JPGs, replay the saved
physical poses without rerunning planning or filtering:

```sh
julia --project=test examples/render_adaptive_log.jl
```

This finds the latest run's `episode.jsonl` under `test/output` and writes
the combined `drone_steps.gif` and `steps/` JPG subfolder in that same run folder. If there are no run folders, it also supports
the older flat `test/output/episode.jsonl`. Optional arguments specify the log
and GIF paths, so an earlier run can be selected explicitly. The animation includes the
pose, decision values, and current fault flags before each executed action,
numbered to match the console decisions, followed by the actual final state.
Log/reward files remain unchanged, and an existing GIF or step-image directory is not overwritten.
Programmatically, `render_saved_adaptive_artifacts(log_path; generate_gif=false)`
can add JPGs to an episode that already has its GIF.

The optional standalone Q-value exporter remains available when specifically
needed; it is not called by the default runner or combined replay:

```sh
julia --startup-file=no --project=test -e 'include("examples/render_adaptive_log.jl"); render_saved_action_value_artifacts("test/output/YOUR_EPISODE/episode.jsonl")'
```

Pass `filename` and `jpg_directory` to that function to select new output paths
when the default chart files already exist.

### Verified 30-step particle run

The exact belief remains the default. For a bounded run, explicitly enable the
particle approximation approved for the 30-step experiment:

```sh
julia --project=. examples/run_adaptive.jl 42 30 --lookahead 1 --particles 128
```

This separates episode H=30, lookahead L=1, and a budget of 128 retained weighted
mode/count histories. A fresh run directory contains the log, reward CSV, SVG
chart and GIF automatically. The seed-42 validation completed all 30 actions,
with total reward -30 and horizon termination; the goal was not reached. It
started at (1,1), froze the left action at step 7, and ended at (2,5), heading east.
A short lookahead with zero terminal value is a planning approximation and does
not guarantee navigation to a distant goal.

The filter first enumerates b*theta*O from the retained support using the SAME
prepared worst-case rows, then samples mode/count histories within configuration
strata. Each configuration receives at least one draw and retains its full
probability mass for this expanded empirical posterior. This preserves the
configuration support, including very small
weights. It does not make the mode/count posterior exact. Counts remain integer
snapshots with one unit increment per transition; resampling never averages phi.
The finite ExactAdaptiveBelief container stores this weighted empirical
approximation when particle mode is enabled. Default runs still enumerate exactly.

`--particles N` is optional, and `--particles=N` is accepted. For this map, N must
be at least 97 to cover every configuration/outcome. `RecedingHorizonRobustPolicy`
also accepts `particles=N, particle_seed=...` programmatically. Its matching
updater can initialize/compress a larger initial distribution. RNG state,
pending model, and episode stage commit only after successful filtering.
The resampling RNG is separate from physical faults, observations and HDR
calibration. Configuration/filter records identify the approximation, budget,
expanded support and retained support. Detailed branch records mark whether a
child survived compression. Larger L still costs more and can reach planning
resource caps.

The final planning layer uses independent HDR row minima directly because its
continuation value is zero. This is the same one-step robust objective, with no
observation-tree expansion. Longer-horizon backups retain their coupled solve.
When fitting final-layer continuations, the planner caches row minima shared by
overlapping supports without constructing or caching another generation of count
histories. This avoids the redundant leaf action caches that could exhaust
`max_total_edges` at L=2 even with 128 retained particles. The particle budget
applies between real actions; hypothetical planning supports can still be larger,
and the remaining planning resource limits still apply.
Training-belief and continuation caches use keys whose types do not depend on
support size, avoiding excessive Julia compilation for large hypothetical beliefs.

With these cache changes, a headless validation of seed 42, H=30, L=2,
128 particles, and `warmup=false` reached the goal at step 8 with total reward 93
under the default resource limits. H=30 is the maximum episode budget; reaching
the goal ends the episode earlier. Reproduce it with the L=2 command above.

### Adaptive successor support and Dirichlet rows

`feasible_successors(pomdp, r, m, a)` returns the ordered feasible `(r′, m′)`
pairs for each of the six actions. For an active source, support is the Cartesian
product of `{stay, commanded configuration}` and `successor_modes(m)`: both
configurations are included for every admissible next mode, regardless of that
action's fault flag. Modes stay unchanged or gain one fault, with no recovery.
Thus nominal mode has 14 pairs and a mode with one existing fault has 12.
The joint probabilities remain unknown; Cartesian support does not impose
probabilistic independence. No sideways slip or unintended rotation is included.

`actions(pomdp, s)` and `actions(pomdp, b)` return all six IDs whenever the
state/belief is active. There is no intersection over hypothetical poses and no
pose-dependent action mask. Outward commands have ordinary learned rows with
both stay and unsafe-entry branches, just like obstacle-directed commands.
Executed boundary crossings enter `UNSAFE_CONFIG`, earn -100, and terminate;
a frozen command stays and costs -1. Terminal states and terminal-only beliefs
have no further decisions. Use the robust policy's matching exact updater; the old DiscreteUpdater
assumes a finite configuration state space and is incompatible with this model.

The support specifies possible outcomes: boundary crossing or obstacle contact maps to
`UNSAFE_CONFIG`, and goal/unsafe states are absorbing. Probabilities come from
explicit selected rows. Physical fault execution is separate from these rows.

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
@assert count_value(pomdp, phi, key) == 2
@assert count_value(pomdp, learned, key) == 3
row = count_row(pomdp, learned, r, m, a)
hdr = build_row_hdr(pomdp, adaptive_state(pomdp, r, m, learned), a; epsilon=0.05)
```

Every feasible transition entry starts at `phi[key] = 2`. The sparse dictionary
stores only visits above that baseline: `phi[key] = 2 + increments[key]`.
One latent transition adds one to its own entry; unvisited entries remain two.
Thus the KKT exponents `phi_row .- 1` are always positive on learned rows.
The symmetric `Dirichlet(2,...,2)` prior has a uniform mean and more concentration
around it than the previous all-one prior; its positive-threshold HDR excludes
simplex vertices. This prior change does not alter structural feasibility.

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

# Explicit demonstration candidate at the symmetric prior's interior mode.
# This is not an estimate of the true dynamics or a worst-case optimization.
theta = fill(1 / length(U.successors), length(U.successors))
physical = locomotion_mode_transition(pomdp, s, a, theta; hdr=U)
rp, mp = rand(MersenneTwister(43), physical)
@assert isempty(s.counts.increments) # sampling this factor never updates phi
```

Goal/unsafe sources have no learned uncertainty row. Use
`locomotion_mode_transition(pomdp, s, a, [1])` without an HDR for their absorbing
configuration/mode distribution. Run the focused tests with
`julia --project=. test/test_locomotion.jl`.

### KKT minimization of one transition row

`minimize_hdr_row(hdr, successor_values)` minimizes `dot(successor_values,theta)`
over the supplied HDR. The values are fixed coefficients in `hdr.successors`
order. They are neither Dirichlet parameters nor a complete belief value
function. The planner below forms them from rewards and continuation values.
This routine implements only that linear row subproblem.

For `d = phi_row - 1 > 0`, it evaluates the closed-form row
`theta[j] = (d[j]/(v[j]+kappa)) / sum(d ./ (v+kappa))` and finds kappa by
bracketed bisection of `logpdf(Dirichlet(phi_row),theta(kappa))-log(hdr.tau)=0`.
This is the slides' equation `sum(d .* log(theta)) - c = 0`, with
`c = log(hdr.tau) + log(B(phi_row))`. Internally, costs are shifted/scaled and
the search stays at a strictly positive distance above `-minimum(v)`.
The existing HDR threshold is reused without sampling or changing counts.

```julia
pomdp = DroneSurveillancePOMDP(obstacles=DSPos[])
s = initial_adaptive_state(pomdp)
a = ACTIONS_DICT[:forward]
hdr = locomotion_mode_uncertainty(pomdp, s, a; epsilon=0.1,
                                 rng=MersenneTwister(42))
# Fixed hand-supplied example coefficients:
successor_values = [rp == s.config ? -1.0 : 9.0 for (rp, mp) in hdr.successors]
result = minimize_hdr_row(hdr, successor_values)
selected = SelectedTransitionRow(result.theta, hdr)
kernel = robust_transition(pomdp, s, a, result.theta; hdr=hdr)
@assert result.theta in hdr
@assert isempty(s.counts.increments)
```

`HDRRowMinimum` returns `theta`, `value`, `kappa`, `lambda`, `shifted_kappa`,
iteration counts, `status`, log-HDR slack, stationarity diagnostics,
`optimality_gap`, and `objective_lower_bound`. The gap uses the convex
Lagrangian and accounts for both complementary slackness and stationarity
error. Diagnostics use floating-point arithmetic rather than interval-certified
bounds. Treat the returned probability vector as read-only.

Constant coefficients return the Dirichlet mode. Singleton and mode-only sets
return their unique point; no kappa is reported on these direct exits. A general
solve returns a feasible endpoint only when the HDR, KKT, and objective-gap
tolerances pass. When Float64 would round kappa onto its forbidden pole, the
reported kappa uses BigFloat; `shifted_kappa` retains the positive distance
without cancellation. The row solve itself still uses Float64.

Learned multi-successor rows require parameters greater than one, as ensured by
the all-two initialization. Unsupported parameters, malformed/empty HDRs, and
invalid values raise `ArgumentError`; numerical limits or nonconvergence raise
`RowMinimizationError`. There is no unverified fallback. Parent counts, map,
and full support are checked when the row is used through the existing model
interfaces. A `SelectedTransitionRow` from this result can be included in the
explicit assignment supplied to the exact belief updater. The finite-horizon
planner below forms continuation coefficients and performs action maximization.

Run the focused checks with `julia --project=. test/test_row_minimization.jl`.

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
and mode probabilities are not factorized. With all-two counts, the nominal
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
the robot's belief using true simulator states. The exact conditional belief
updater and robust planner are described below; the augmented-state POMDP wrapper
remains separate work. `POMDPs.transition` on the existing DSState environment retains its original
behavior. Run kernel tests with
`julia --project=. test/test_adaptive_transition.jl`.

### Exact belief update conditional on selected rows

`ExactAdaptiveBelief(pomdp)` starts with the known initial configuration,
nominal mode, and prior counts. For an uncertain initial condition, supply
`ExactAdaptiveBelief(pomdp, [s1 => w1, s2 => w2, ...])`, where each state is a
complete `AdaptiveState`. Duplicate full hyperstates are merged and weights
are normalized. Each stored hypothesis has its own immutable count snapshot;
`support(b)` and iteration return independent copies. `pdf(b,s)` and
`logpdf(b,s)` expose its probability and log probability.

Filtering requires an explicitly selected transition row for **each active
parent**. The planner must choose these rows for the current belief and action
before the resulting real observation. There is no implicit posterior-mean or
uniform fallback, and the updater does not implement the worst-case argmin.
For each parent `s`, wrap the planner's chosen `theta_s` and corresponding
`hdr_s` in `SelectedTransitionRow(theta_s, hdr_s)`. Its probabilities follow
`hdr_s.successors` order. Then prepare the assignment:

```julia
b = ExactAdaptiveBelief(pomdp)
updater = ExactAdaptiveUpdater(pomdp)

# Before executing a: chosen_rows is a Dict mapping every active parent s
# to SelectedTransitionRow(theta_s, hdr_s), supplied by the planner.
selected = BeliefTransitionRows(pomdp, b, a, chosen_rows)
prepare_belief_update!(updater, b, a, selected)

# After executing a and receiving camera observation o:
b_next = update(updater, b, a, o)
```

The assignment validates each parent's feasible successors, count row, and HDR,
and is bound to the full prior belief and action. It snapshots caller-owned
rows and counts. `update` consumes the prepared assignment on success; missing,
stale, or reused assignments fail explicitly. Terminal parents need no row.
`RobustPolicy` prepares this assignment during action selection, and its matching
updater advances the decision stage after a successful observation update.
Connecting it to the physical fault simulator is still pending.

The core `exact_belief_update(pomdp,b,a,o,selected)` can also be called directly,
including when evaluating hypothetical planning observations. For every
latent successor it accumulates
`b(s) * theta_s[j] * O(o | rp,mp,a)` into the full child hyperstate with
`phi_prime = phi + delta(r,m,a,rp,mp)`. It uses the existing locomotion and
parameter-update factors. Identical complete children merge; different count
histories remain separate. The new belief replaces the previous one. Already
terminal hypotheses absorb without increasing counts.

Weights and camera likelihoods are computed in log space without threshold
pruning or particle sampling. Extremely small hypotheses remain in the stored
support even if their `pdf` view underflows; use `logpdf` to inspect them.
Zero observation evidence raises `ZeroObservationEvidence` rather than resetting
the belief. `actions(pomdp,b)` returns all six commands regardless of tiny pose
hypotheses; risk is evaluated in each action's robust backup. The policy and updater receive no true hidden
mode, count history, or fault schedule.

This is an exact enumeration **conditional on the supplied model**; it becomes
the agreed pessimistic filtering convention when the argmin supplies the
worst-case rows. Selecting those rows does not reveal the true dynamics.
Use `return_info=true` on the core to obtain `belief`, `log_evidence`, `parents`,
and `expanded_branches`. Support can grow quickly; no approximation is inserted
to limit that growth. The focused checks reproduce the worked 14- and
172-hyperstate examples with explicitly chosen uniform admissible rows:
`julia --project=. test/test_exact_belief.jl`.

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

Feasible unsafe entries include boundary crossings and obstacle collisions and
receive -100. A stay before a boundary or obstacle receives -1,
since no collision occurred. Inputs must be canonical and form a feasible
configuration/mode transition. The AR module has no legacy occupancy reward. Run reward tests with `julia --project=. test/test_adaptive_reward.jl`.

### Finite-horizon robust value and action selection

The planner retains the paper's Bellman layout, with current time n and terminal
layer N:

```text
V_n(b) = max_a min_Theta sum_s b(s) [r_Theta(s,a)
                 + gamma * sum_(t,z) p_Theta(t,z|s,a) V_(n+1)(b')],  V_N=0.
r_Theta(s,a) = sum_t T_Theta(t|s,a) R(s,a,t)
p_Theta(t,z|s,a) = T_Theta(t|s,a) O(z|t,a)
```

Expected current reward and future value use the same selected transition
assignment. Terminal beliefs have value zero. The horizon/stage is external to
phi; entering the goal/collision increments the transition count once and later
absorbing transitions preserve it.

`RobustValueFunction` stores finite continuation collections for each stage and
complete hyperstate support. `value(vf,b,n)` evaluates their maximum weighted
value. Coefficients are indexed by full `(r,m,phi)` snapshots. All six actions
are evaluated over every hypothesis, including tiny masses, with boundary risk
represented by unsafe transition rewards rather than action exclusion.
Since learned HDR rows have positive probabilities, each observation's support
is fixed during a backup; camera likelihood zeros determine its structural mask.
Log likelihoods preserve this support even when probability views underflow.

This is a **finite continuation approximation for planning**. The belief updater
still performs exact enumeration. Every layer starts from a valid always-rotate
policy lower bound, costing -1 per remaining action on active hypotheses. Last
decision layers consider every admissible action and are complete up to the row
solver's numerical tolerance. Earlier layers improve at representative beliefs
through backward point-based fitting; no optimality claim is made for the full
unrepresented belief space.

```julia
using POMDPs: value, action, updater, update

pomdp = DroneSurveillancePOMDP(region_A=DSPos(4,5), obstacles=[DSPos(3,5)])
b = ExactAdaptiveBelief(pomdp)
vf = RobustValueFunction(pomdp; horizon=2, epsilon=0.1, n_samples=512, seed=42)
report = fit_value_function!(vf, b) # planning calculations, no physical episode
println(value(vf,b,0))
decision = robust_decision(pomdp,b,0,vf)
println(decision) # action, value, selection gap

policy = RobustPolicy(vf)
belief_updater = updater(policy)
a = action(policy,b) # also prepares the chosen model before the observation
# After the physical environment produces observation o:
# b_next = update(belief_updater,b,a,o)
# policy.stage is then 1; phi is not used as a clock.
```

For `fit_value_function!(vf,b; expand=true)`, explicit HDR-mode rows generate
representative training beliefs. This is a point-generation choice, not a
transition fallback in the optimization or real filter. Every structurally
possible observation is enumerated with the pure exact updater. Alternatively,
supply explicit points as `Dict(0 => [b0], 1 => [b1,b2], ...)`. Additional sweeps
can enlarge earlier continuation collections. Componentwise dominated vectors on
the same support/stage are removed without changing that collection's value.

`backup_action(pomdp,b,a,n,vf)` freezes the next continuation collections and
calibrated HDRs, then solves the action's inner minimum. Observation-specific
dual mixtures form fixed row coefficients
`v[s,j] = R(s,a,t_j) + gamma*sum_z O(z|t_j,a)*alpha_bar[z](t_j)`.
The existing KKT/root routine minimizes every active row. Projected dual updates,
feasible primal averaging, and a fully corrective finite-mixture master improve
the saddle solution. The restricted master's log-barrier Newton calculation
only mixes HDR-feasible row assignments; transition rows still come from KKT
oracles. An explicit primal-dual gap, including row-oracle errors, controls
acceptance. Failed solves raise `RobustPlanningError`; no unchecked model is
returned.

Future belief values are evaluated using unnormalized joint masses:
`P(z)*max_alpha sum_t b'(t)*alpha(t) = max_alpha sum_(s,t) b(s)*T(t|s,a)*O(z|t,a)*alpha(t)`.
This identity cancels the posterior normalization during optimization. It does
not remove the observation model or replace hidden hypotheses with a known next
state. Tests independently reconstruct the posteriors with `exact_belief_update`
and compare against this paper-layout objective.

`RobustActionBackup` reports `current_reward`, `expected_future_value`, bounds,
gap, iteration count, a supporting continuation vector, and `selected_rows`.
Its feasible upper value equals current reward plus discounted future value,
up to roundoff. `value` is its lower bound. `robust_decision` compares all
available actions and verifies the resulting selection gap. By default, stable
action-ID order resolves machine-precision ties; `tied_action_rdn=true` samples
uniformly among these ties. **Bounds concern the inner problem for
the specified finite continuation collection and calibrated HDRs**, not a global
certificate for the true dynamics or the exact infinite collection. Arithmetic
and diagnostics are floating-point, not interval certified.

`prepare_decision!` can prepare a decision in an `ExactAdaptiveUpdater` directly.
`RobustPolicy` and its matching `updater(policy)` automate that handoff: repeated
action queries reuse the pending model, and errors leave the stage unchanged.
Successful observation updates consume the model and advance n by one.
`reset_robust_policy!` or matching-updater initialization resets an episode.
No true simulator state, mode, count history, or fault schedule is an input.
Neither fitting nor hypothetical belief updates consume a real pending model.

`RobustPlannerOptions` exposes the gap tolerance, iteration limit, and caps on
supports, hypotheses, cached actions/rows/edges, score entries, mixture columns,
continuation vectors, and training points. Exceeding a cap reports an error;
it does not prune hypotheses, silently shorten N, or substitute a mean row.
Model/calibration edits invalidate the caches and require a new value function.
`RobustPolicy` retains the configured finite horizon. The example now explicitly
uses a separate `RecedingHorizonRobustPolicy` to separate episode H from local L.

Run `julia --project=. test/test_robust_planning.jl` for focused planner checks.
Physical fault integration is implemented below; the requested review and final
example run remain pending.

### Independent episode horizon and receding lookahead

`RecedingHorizonRobustPolicy(template; H)` uses `template.horizon` as L. Its
`stage` counts completed episode actions. Before each new decision it builds and
fits a fresh local value function with horizon `min(L,H-stage)` from the current
exact belief. Planning starts at local stage zero; the episode clock is not reset.
Repeated action queries for a pending belief reuse the model without refitting.
After an observation, its matching updater uses that model, retains every branch's
updated phi, and advances the episode clock once. Failed filtering preserves both
the pending model and clock. Reset/initialization explicitly starts a new episode.

The original `RobustPolicy` remains available for full finite-horizon experiments.
Changing H in the example no longer changes L. Training-point selection, exact
belief enumeration, the zero local terminal layer, and all resource limits remain
as before. This change alone does not solve long-horizon belief-support growth.

### Augmented simulation termination

`POMDPs.isterminal(pomdp, s::AdaptiveState)` recognizes goal and collision
outcomes. `termination_reason(pomdp, s, steps, H)` returns `:goal`, `:collision`,
`:horizon`, or `nothing`. Here `steps` counts completed actions, including
staying and rotating. Goal/collision take precedence on step H. The horizon is
external to the hyperstate and count keys; it does not create a physical
terminal state or an extra reward.

The robust entry point is `simulate_robust(pomdp, policy; ...)`. It checks that H
matches the policy's remaining episode budget (the remaining planning horizon for
finite RobustPolicy, or independent episode budget for the receding policy),
obtains `updater(policy)`, and wires a
hidden physical environment to `simulate_adaptive`. The lower-level runner also
accepts an explicit custom transition kernel for component fixtures. Neither
entry point substitutes posterior-predictive rows for the selected robust model.

The result contains `history` (POMDPTools.SimHistory), `final_state`,
`final_belief`, `reason`, `steps`, `total_reward`, `terminated`, and `truncated`.
History includes physical states and actual fault events for evaluation, alongside
action, observation, reward, beliefs before/after, and decision summary. Final
entry rewards/counts are retained. H=0 and initially terminal states execute no
actions. Rendering keeps the drone visible at its goal configuration.

### Integrated execution, learning, and logs

```julia
pomdp = DroneSurveillancePOMDP(region_A=DSPos(1,1), obstacles=[DSPos(3,5)])
b0 = ExactAdaptiveBelief(pomdp)
template = RobustValueFunction(pomdp; horizon=2, epsilon=0.1, n_samples=512, seed=42)
policy = RecedingHorizonRobustPolicy(template; H=3) # episode H=3, lookahead L=2
# Run only after the requested main-module review:
result = simulate_robust(pomdp, policy; H=3, N_f=2, seed=42,
    initial_belief=b0, log_path="/tmp/my_ar_run/episode.jsonl",
    reward_csv_path="/tmp/my_ar_run/rewards.csv",
    reward_chart_path="/tmp/my_ar_run/rewards.svg")
```

Each step selects an action and complete worst-case row assignment using the
parent counts. Those rows are prepared before the observation. The environment
then activates any scheduled fault, executes motion, and generates reward and
camera data. The exact updater uses that SAME assignment to weight branches by
b*theta*O; each branch carries its own unit count increment. Updated branch counts
determine the HDRs at the next decision. Only the executed source/action row is
incremented; other outgoing rows may still have all-two parameters. There is no
single global count estimate and no physical count history passed to filtering.

Terminal output distinguishes episode stage, local planning stage and effective
lookahead, and includes local fit sizes, action values, selection gap, a representative phi/theta
row, camera labels, reward, cumulative reward, belief size, highest-weight
hypotheses, mode probabilities, and stop reason. Physical position/mode/faults and
the realized count change are explicitly labeled simulator truth. They are not
inputs to planning or filtering.

The flushed JSONL log contains configuration and map/calibration settings,
episode H, configured L and planning mode, pre-observation decisions with effective
lookahead and episode/local stages, full selected rows, physical transitions, count-update
branches, posterior hypotheses/marginals, and termination/error records. Stable
hypothesis and row IDs connect these entries. `detailed_log=false` omits full rows,
hypotheses and branch detail; `top_k` limits displayed hypotheses only, without
pruning the actual belief. `io=nothing` silences terminal output. An error stops
the episode and preserves partial log/CSV output; it does not reset the belief
or substitute a model. Do not retry a physically executed step as if it had not
occurred.

`rewards.csv` records executed step and immediate transition reward. The vector
line chart `rewards.svg` uses those same rewards against step (not cumulative
return or planner value), including the final goal/collision reward. Empty and
single-step histories are supported. `save_reward_chart(path, history)` can also
plot a saved/in-memory history; Compose's SVG backend needs no extra dependency.

Focused integration fixtures: `julia --project=. test/test_adaptive_integration.jl`.
The final fitted example and long-horizon benchmark have not been run.

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

`HiddenFaultEnvironment` connects this schedule to execution. At each event it
chooses a previously healthy action uniformly and freezes it before executing
that step's selected command. The action remains frozen for the episode and
remains selectable. Unexecuted events after early termination do not occur.
A healthy command performs its intended motion/rotation; a frozen command stays.
Healthy outward commands enter the unsafe terminal state; frozen commands stay.
All six commands remain available. This physical execution rule never
prunes the learner's Cartesian support.

The resulting count key uses the original (r,m,a) and final (r',m'), including a
newly activated fault in m'. Fault activation and motion produce ONE joint count
increment. `simulate_robust` separates RNG streams for occurrence times, healthy
action choices, and camera observations; HDR calibration has its own seed.

For controlled tests, supply `fault_steps=[1]` and `fault_actions=[4]` to freeze
right on step 1. Explicit steps must be sorted/distinct and actions must be
initially healthy/distinct; the schedule must respect H and N_f. With an initially
faulty physical state, N_f caps the total and random sampling uses its remaining
budget. Simulator schedules/actions never enter the policy or belief updater.
Run scheduling tests with `julia --project=. test/test_fault_schedule.jl`.

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
