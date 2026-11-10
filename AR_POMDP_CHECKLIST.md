# AR-POMDP implementation and final-run checklist

## Fixed-action follow-up (2026-11-03)

The user replaced pose-dependent action availability with all six commands at
every nonterminal state/belief. Historical boundary-rejection/intersection notes
below describe the earlier model. Executed boundary crossings now share the
unsafe collision outcome and -100 reward; frozen commands stay for -1. Learned
rows, count keys, filtering, physical execution and planning support this rule.
Console diagnostics report boundary risk without action vetoes. GIF/JPG frames
retain pre-action poses but use simple "Step N" titles. Full suite: 73,863 passing
assertions. Verified seed42/H30/L1/particles128 output is
test/output/run_seed42_H30_L1_particles128_D1ReXs/; all six actions are evaluated
at all 30 decisions, although the myopic planner still selects rotations.

This records the user's requested review points and progress on the original,
exact belief updater. Focused component checks do not replace the pre-run review
or authorize the final integrated simulation/test run.

## Review before finalizing and running the integrated simulation

- [x] Remind the user to review `src/DroneSurveillanceAdaptive.jl` before the final run.
- [ ] Review the agreed filtering convention: the belief update is conditional
  on the selected admissible worst-case transition model. It is a pessimistic,
  model-conditioned update; selecting a model does not reveal the true dynamics.
- [ ] Use the same filtering convention when the argmin evaluates future belief
  values. Do not silently substitute posterior-predictive or uniform rows.
- [ ] Reuse the row assignment chosen for the current belief/action before the
  resulting real observation. Do not choose a new adversarial row using that
  observation or assume the selection is observable to the robot.
- [x] Integrate AdaptiveState = (r, m, phi) throughout the simulation and migrate
  the old DSState model interfaces before removing their dependencies. Retain
  DSState as the configuration component r where needed.
- [x] Connect hidden scheduled fault activation without exposing the true mode,
  counts, or schedule to the policy or belief updater.
- [x] Preserve learning order: select the current model using each parent's
  original phi; filter with that same model and the real observation; increment
  each feasible successor branch's own transition coordinate; use its updated
  phi to construct/reuse the corresponding Dirichlet HDR at the next decision.
  Do not replace the belief's distinct count histories with one global count row.
- [x] Add readable per-step terminal output and a persistent structured log.
  Include action/stage, candidate action values and selection gap, selected-row
  summaries, observation/reward, belief size and configuration/mode marginals,
  representative count changes, and termination. Include full selected rows and
  branch count keys in detailed logs, with configurable verbosity.
  Label simulator truth (physical configuration, faults, mode, realized count
  key) separately for evaluation; never pass those diagnostics to planning or
  filtering. Use stable row/hypothesis identifiers to relate before/after data.
- [x] Implement a reward-versus-step line chart from the recorded episode history:
  x-axis = executed step (1 through the actual stopping step), y-axis = immediate
  transition reward R(s_t, a_t, s_{t+1}). Save the chart beside the episode log and
  support empty/single-step histories. Do not substitute cumulative reward or
  planner value; use the same recorded rewards as the terminal/log output.

## Exact belief updater implementation

- [x] Initialize every feasible Dirichlet coordinate at two; store `phi - 2`
  sparsely, increment the selected coordinate by one, and keep all learned-row
  KKT exponents `phi - 1` positive. Structural zeros stay excluded.
- [x] Add `src/adaptive/belief.jl` with a finite weighted belief over complete
  AdaptiveState hypotheses. Protect count snapshots from mutation, including
  mutation through caller-owned inputs or dictionary keys.
- [x] Initialize a known start with one nominal hyperstate and its prior counts;
  support explicitly supplied finite initial beliefs as well.
- [x] Implement the core update conditional on an explicit assignment of rows to
  parent hypotheses. Keep row selection outside this core so an argmin can
  evaluate candidate posteriors without recursively invoking itself.
- [x] Require a complete row assignment for positive-mass active parents. Validate
  each row against that parent's full feasible support, own counts, and HDR.
  Existing terminal parents use their absorbing transition without a learned HDR.
- [x] Expand all positive-mass feasible successors, apply the known observation
  likelihood, obtain each branch's deterministic parameter update exactly once,
  and accumulate probability contributions by the complete next hyperstate.
- [x] Merge identical (r, m, phi, derived outcome) hypotheses; retain different
  count histories even when position and mode agree. Replace the previous belief
  with the new one rather than appending old hyperstates.
- [x] Normalize stably. Report zero observation evidence explicitly; do not
  silently reset the belief, prune small weights, or introduce particle resampling.
- [x] Add POMDPs initialization/update adapters and a belief-dependent boundary
  action interface. Reject unavailable actions without dropping incompatible
  parent hypotheses to force a successful update.
- [x] Include/export the new interfaces from `src/DroneSurveillanceAdaptive.jl`.
  Use the existing simulation updater hook during the later integration phase.

## Objective and planner implementation

- [x] Keep the scalar robust belief value `V_n(b)` separate from Dirichlet row
  parameters (`phi_row`) and the fixed successor coefficients (`successor_values`)
  passed to the KKT minimizer. Finite continuation collections and backward
  point-based fitting are implemented as an explicit planning approximation;
  the belief updater remains exact enumeration.
- [x] Implement and validate KKT/root minimization on the all-two learned rows,
  including constant costs and degenerate HDRs, before the full robust backup.
  `minimize_hdr_row(hdr, successor_values)` returns the minimizing row and
  multipliers with explicit HDR/KKT/objective-gap diagnostics. The threshold and
  count snapshots remain unchanged; invalid or failed solves raise errors.
- [x] Couple continuation choices and transition rows in the robust backup;
  preserve boundary action eligibility and return the selected model to filtering.
- [x] Preserve the paper's layout and current-time index: V_N=0, current reward
  r_Theta=sum(T_Theta*R), and joint p_Theta=T_Theta*O with the same chosen model.
- [x] Compare all valid actions with an inner/selection gap; retain the selected
  primal model before real observations. No mean-row fallback or true-state input.
- [x] Add RobustPolicy and its updater, with stage advancement only after a
  successful belief update and an explicit episode reset.
- [x] Connect the robust policy and matching exact updater to hidden physical
  faults in simulate_robust. The final fitted example remains pending review.
- [ ] Benchmark larger planning horizons and training collections; distinguish
  finite-continuation approximation from the exact full robust value function.

## Validation and later performance work

Focused checks passed after the all-two prior change: 90 exact-belief assertions,
69,592 component assertions for successors, boundary actions, HDRs, locomotion,
parameter updates, combined transitions, rewards, and fault scheduling, 284
termination-fixture assertions, and 30,802 legacy HDR visualization math checks.
The visualization checks did not generate new figures. No final integrated
AR simulation was run.

The row minimizer additionally passes 299 focused assertions, including analytic
solutions, log-residual equivalence to the slides, learned rows, ordering,
precision cases, actual reward coefficients, and transition/belief integration.
After adding the solver, the HDR, locomotion, parameter-update, combined-kernel,
and exact-belief regressions also pass (33,966 assertions).

The paper-layout planner passes 151 focused assertions, covering one-step
reward/maximization, two-step observation-dependent continuation values checked
against an independent probability-grid calculation, backward fitting, distinct
count histories, terminal/horizon and boundary rules, tiny probability support,
cache invalidation/resource failures, and the policy/updater stage lifecycle.
The broader HDR/transition/reward/belief/row regressions pass (67,559 assertions).
The documented two-step planning example was checked without a physical episode:
29 training beliefs, action 4, value about 23.188677, and selection gap below 2e-10.

- [x] Check hand-calculated one-step posterior weights and count snapshots.
- [x] Reproduce the 14- and 172-hyperstate examples using their explicitly chosen
  uniform admissible rows, map, actions, and observations. These are conditional
  examples, not fixed support sizes for arbitrary rows or observations.
- [x] Check merging, distinct histories, parent/sibling independence, HDR row/count
  validation, terminal absorption, boundary validity, and zero evidence.
- [x] Check that the updater receives only belief, action, observation, and the
  explicit selected model; never the actual environment hyperstate or schedule.
- [ ] Warm up Julia before measuring update time, allocations, parent count,
  expanded branches, and unique posterior support on short sequences.
- [ ] Increase sequence length gradually. Stop and report resource limits instead
  of silently approximating an exact belief.
- [ ] Complete the pre-run review above before the final integrated run.

## Physical integration implementation and review handoff

- [x] Declare POMDP{AdaptiveState,Int64,DSObservation}, initialize nominal/all-two
  hyperstates, and split model-independent types from map-aware constructors.
  Remove the finite configuration-only state enumeration, old DSState transition,
  occupancy reward, and terminal-state field from the AR module. Keep DSState as r.
- [x] Execute hidden faults using independent schedule/action/observation RNGs;
  preserve source mode in the one joint transition count increment. Keep frozen
  actions available and learner Cartesian support unchanged.
- [x] Replace the forced-route visualization with a guarded robust example.
  The original finite policy has N=H. The example now explicitly uses independent
  episode H and receding lookahead L; neither policy uses a mean-model fallback.
- [x] Add JSONL decisions/selected rows/branch count changes/posterior records,
  labeled physical truth, terminal summaries, reward CSV and an SVG line chart.
  Detailed output is configurable; display limits do not prune exact beliefs.
- [x] Check controlled one-/two-step integration fixtures, all six frozen actions,
  delayed faults, collision/stay differences, boundaries, count keys, log/CSV/chart
  output, horizon synchronization, and reuse of the original selected rows.
  831 focused assertions pass, including the additional initially-faulty capacity
  case. The complete repository suite passed 71,616 assertions before those 20
  additional focused checks; the affected integration suite was rerun afterward.
- [x] Check that the guarded visualization entry point and optional GIF
  dependencies load without fitting, running an episode, or writing a GIF.
- [x] Render and inspect the chart using synthetic fixture rewards, without
  presenting those values as an actual episode result.
- [ ] User review of src/DroneSurveillanceAdaptive.jl before the final fitted run.
- [ ] Run the reviewed example and show its actual log, reward CSV/chart and GIF
  if requested. No final fitted example or long-horizon benchmark has run yet.

Entry point: examples/run_adaptive.jl [seed=42] [H=2] [output_root] [N_f=2] [--lookahead L=2].
Each run creates a persistent unique run_seed<seed>_H<H>_L<L>_<suffix> folder inside
/home/ysl/DroneSurveillance.jl/test/output, resolved from the script location.
An explicit third argument changes the parent output root. Logs, reward CSV,
chart, and GIF share the new run folder; previous output is preserved. The main
example now renders the GIF by default, with generate_gif=false available for
programmatic headless runs. Empty episodes skip animation. Graphics dependencies
are declared in the main project; the old visualization script delegates once
to the main example instead of duplicating rendering.
The GIF replay defaults to the latest run log, with legacy flat-log fallback.
The example now starts at (1,1), with goal (5,5) and obstacle (3,5), as requested.
H=2 remains a short check; reaching the goal needs at least eight successful moves.
A broader map and larger H must be explicit; the initial example is deliberately
small because exact support and finite-continuation planning can grow quickly.

## Saved episode GIF

The user subsequently generated test/output/episode.jsonl, rewards.csv and
rewards.svg. That saved episode used the previous start (4,5) and reached the goal
after one action, reward +100. It has not been regenerated with the new start.
Generated test/output/drone.gif from those logged physical poses with
examples/render_adaptive_log.jl (no planning, simulation, or filtering rerun).
Verified two distinct frames (initial pose plus goal pose) and unchanged SHA-256
hashes for all three original output files. New GIF runs also include the initial
pose. JSON is declared in the optional visualization/test environment for replay.

The main examples/run_adaptive.jl entry point now renders drone.gif automatically
alongside each run's log/CSV/chart. A one-step check from (1,1) verified all four
files and a valid GIF header using the main project environment. The compatibility
visualization entry point delegates to it, so the GIF is rendered only once.
Also replayed the user's two-step run_seed42_H2_P8giZM log to create its missing
three-frame GIF, without rerunning the episode.

## Independent episode budget and planning lookahead

- [x] Keep finite RobustPolicy behavior; add explicit RecedingHorizonRobustPolicy.
- [x] Separate episode H/stage from local horizon min(L,H-stage)/stage zero.
  Fit from the current exact belief for each new action; reuse pending selections
  for repeated queries and real filtering, preserving all learned count histories.
- [x] Use episode H for physical fault scheduling and termination.
- [x] Add --lookahead L to the example without changing its positional arguments.
  Default L=2 is independent of H; log H, L, effective window and both clocks.
- [ ] Improve training-belief selection and exact-belief scalability separately.
  Local terminal value is still zero: short lookahead can miss distant goal reward.
  This change alone does not guarantee completion of a 30-step exact-belief run.

H/L separation validation: 143 focused assertions and 71,779 full-suite assertions
passed. Removed redundant final-layer point backups, cached identical observation
support masks, and moved support-limit checks before sorting/tuple construction
and HDR calibration; an additional focused preflight check passed.

Attempted H=30, L=2, seed 42 from (1,1). A completed diagnostic attempt stopped
before action 3 because its next exact support exceeded max_hypotheses=5000.
Recorded posterior support 1 -> 14 -> 172; independent successor enumeration
gives 1,832 then 16,720 for the next two forward actions. See
test/output/run_seed42_H30_L2_mAtCQI/{run_report.md,console.log,episode.jsonl,rewards.csv}.
Other development retries were interrupted during performance investigation;
none is a completed 30-step episode. User approval for a particle belief
approximation is pending. Exact belief remains unchanged; no 30-step completion
or general bug-free guarantee is claimed.

## Completed 30-step validation (particle mode approved by the user)

The user explicitly approved particle approximation after the exact run hit its
resource bound. Added optional --particles N / policy particles=N; exact mode
remains the default. Mode/count histories are sampled within configuration
strata, preserving every configuration and its current empirical posterior mass
for safe boundary selection. Phi snapshots retain integer unit increments.
The matching updater commits the pending model, sampling RNG and clock atomically.

Optimized the final planning layer to use its exact row-wise immediate-reward
minima directly (V_N=0), avoiding unnecessary observation-tree construction.
The general multi-step coupled backup remains unchanged.

Completed command: examples/run_adaptive.jl 42 30 --lookahead 1 --particles 128.
Output: test/output/run_seed42_H30_L1_particles128_NlVmoJ/.
Result: 30 executed actions, horizon termination, total reward -30, no collision,
start (1,1), final position (2,5), heading 90. Left action froze at step 7.
The goal was not reached; no optimal-navigation guarantee is claimed.

Verified all 30 selected-row assignments satisfy their logged HDR thresholds,
all true and retained-history count updates are unit/integer, retained support
never exceeds 128 (maximum observed 122), faults obey the cap, CSV rewards match
the history, and the GIF has 31 frames. The full repository test suite passed
71,813 assertions, including 143 horizon and 34 particle assertions.
No general bug-free guarantee is inferred from these checks.

## Computation timing

- [x] Time planning, physical simulation/observations/rewards, and belief update
  separately from diagnostics and rendering; include GC within those phases.
- [x] Record per-step timings and the episode total in JSONL and terminal output.
- [x] Save timing.json in each new example output folder before rendering the GIF.
- [x] Warm up the same seeded episode, then use fresh counts, caches and RNGs for
  the measured episode. Report residual Julia compilation separately and subtract
  it once from core phase wall times (Julia >= 1.11); label this an estimate.
- [x] Distinguish raw core time, compilation-adjusted computation time, and episode
  wall time including logging. Earlier run folders have no recoverable exact timing.

Verified the timed 30-step example in
test/output/run_seed42_H30_L1_particles128_C36Li0/: core computation 1.132345977 s,
residual core compilation 0 s, loop wall time with diagnostics 2.471542010 s.
The physical trace matches the previous NlVmoJ run; timing arithmetic and all 30
step events agree. GIF rendering follows timing output. Focused integration,
receding-horizon, particle and timing suites passed 1,042 assertions (34 new).
