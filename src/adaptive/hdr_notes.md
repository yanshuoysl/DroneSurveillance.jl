# HDR implementation notes

The first version samples from each transition row's Dirichlet distribution,
evaluates PDF values, and estimates the lower epsilon quantile with Julia's
`Statistics.quantile`, using default linear interpolation. The resulting set is
the continuous density superlevel set on that row's simplex.

Reference: [Hyndman (1996), Section 3.2, author PDF pages 7–8](https://robjhyndman.com/papers/hdrgraph.pdf#page=7).
The author PDF says “jth largest” while describing the j/n lower quantile;
the lower quantile is consistent with retaining probability at least 1-epsilon.
Julia's default interpolation is the explicitly selected convention for this
implementation; it is not the paper's particular floor(K*epsilon) selection.

## Possible later experiments

- Evaluate log densities and store log(tau) to handle large or small PDF values.
- Compare order-statistic conventions, including floor(K*epsilon) and the
  previously proposed floor(K*epsilon)+1 empirical-retention rule.
- Estimate Monte Carlo threshold error and investigate adaptive sample budgets.

These alternatives are not implemented in this version. PDF overflow or underflow
at calibration samples causes an explicit error; there is no log-space fallback.
All-one counts give a constant density and the whole simplex under the stated
superlevel-set definition. A single feasible destination gives the singleton (1).

## Action-dependent successor support

`feasible_successors(pomdp, r, m, a)` now supplies the complete, consistently
ordered support S(r,m,a) for all six actions. For an active source it is the
Cartesian product `{stay, commanded configuration} × M(m)`: both configurations
are paired with every admissible next mode, even when that action's fault flag
is set. Modes remain unchanged or gain one fault. Fault flags do not select a
motion outcome deterministically; probabilities over the joint pairs remain
unknown. Cartesian support does not impose probabilistic independence.
For six flags and k existing faults, a row has `2 * (7-k)` entries: 14 for nominal
and 12 for one existing fault. Ordering lists all modes for staying, then all
modes for commanded motion. Translations retain heading and rotations retain
position. No slip or unintended rotation is included. Outward commands are
excluded from state-dependent action sets and rejected by model transition and
count APIs; they have no row or stay branch. Availability never consults hidden
fault flags. Obstacle contact becomes `UNSAFE_CONFIG`; the legacy DSState
simulator still blocks obstacle entries.
Goal and unsafe states are absorbing and have no learned count rows.

`is_feasible_successor` returns false for impossible pairs. These pairs are
structural zeros: exclude them from the Dirichlet row rather than assigning
them a zero concentration parameter or an implicit prior count of one.

Use `CountKey(pomdp, ...)`, `count_value(pomdp, phi, key)`,
`increment_count(pomdp, phi, key)`, and `count_row(pomdp, phi, r, m, a)` for
validated model counts. An explicitly ordered map-aware row must contain all
feasible pairs exactly once. Different actions always index separate counts.
The original overloads without a map remain low-level storage primitives for
illustrative snapshots; they cannot establish physical feasibility.

`build_row_hdr(pomdp, s, a; epsilon=...)` automatically uses the validated
support. Its older overload accepting an explicit successor list remains a
low-level fixture interface and does not check locomotion/completeness. In
particular, the 140-entry Cartesian neighborhood used by the visualization is
an illustration, not an action-dependent transition row.

## Locomotion and mode dynamics

The AR-POMDP's first factor is represented separately in `adaptive/locomotion.jl`.
`locomotion_mode_uncertainty(pomdp, s, a; epsilon=...)` follows the flow
`(r,m,phi,a) -> S(r,m,a) -> alpha -> U(phi)` and returns a DirichletHDR, not a
posterior-mean or sampled row. For a supplied theta in this set,
`locomotion_mode_transition(pomdp, s, a, theta; hdr)` returns a categorical
distribution over `(r',m')` only, with mass theta[j] on the jth feasible pair.
This factor reads phi without modifying it, never constructs phi', and performs
no parameter update even when its distribution is sampled.

An active source requires an explicit theta and matching HDR. There is no
posterior-mean fallback. Matching checks enforce full successor support and
current row counts; membership uses the existing calibrated threshold without
new Monte Carlo samples. Threshold calibration retains approximate coverage.
Worst-case row selection still belongs in the planner. Goal/unsafe sources use
the known absorbing pair distribution with theta=[1] and no uncertainty set.

## Parameter-update transition

The second factor is implemented separately in `adaptive/parameter_update.jl`.
`parameter_update_transition(pomdp, s, a, rp, mp)` returns a Deterministic
distribution over `phi + delta^a_{(r,m),(rp,mp)}`. Its PDF equals 1 for exactly
this full count vector, and 0 for every other one. Only the selected joint
transition count changes; the sparse representation stores phi-1, so its entry
also increases by one. Input counts and sibling histories are preserved.

The interface checks that the conditioning pair is feasible, including map,
action, and irreversible mode constraints. It does not use theta, an HDR,
posterior means, or observations. Stay branches and entry into terminal outcomes
increment once. Already terminal sources only allow their absorbing pair and
return a separate snapshot of unchanged counts.

Conditioning on a latent transition makes the parameter update deterministic;
the robot can still be uncertain over which history and count vector occurred.
The belief updater must retain that uncertainty rather than being given the
true transition from the simulator.

## Combined transition kernels

For s=(r,m,phi) and j=(r',m') in S(r,m,a), the candidate transition kernel is
`T_theta(s'|s,a) = theta[j]` when `s'.counts = phi + delta(r,m,a,r',m')`, and zero
otherwise. As a Bayesian baseline, `adaptive_transition(pomdp, s, a)` uses the Dirichlet posterior
predictive probabilities `phi_row / sum(phi_row)`. The overload accepting an
explicit theta checks its simplex membership and uses the same deterministic
count update. Support remains Cartesian; fault flags never prune motion outcomes.

`robust_transition(pomdp, s, a, theta; hdr)` now obtains its physical pair
distribution from `locomotion_mode_transition`, then obtains each branch's
count vector from `parameter_update_transition`. Thus the full kernel composes
the two separate factors in the formulation. The first factor checks HDR membership.
The HDR must match the full support and the current row counts, in the supplied
successor order. This prevents using a stale threshold after a count update or
using a reduced illustrative fixture as a model uncertainty set. Selection of
the worst-case theta depends on continuation values and belongs in the planner.
No Monte Carlo calibration or adversary optimization runs inside this kernel.

On every branch, only that joint transition's count is incremented. Each next
hyperstate has its own snapshot, including stay branches and entry into terminal
outcomes. Goal/unsafe sources are absorbing with unchanged mode and counts;
they have no learned row. These are conditional latent-state transitions, not a
belief updater that can access the true mode or infer a unique observed history.

## Transition reward

`adaptive_reward(pomdp, s, a, sp)` (also available as the four-argument
`POMDPs.reward` method for AdaptiveState) gives +100 on goal entry, -100 on
obstacle collision, and -1 otherwise. Terminal-source absorbing transitions
give 0, so goal/collision rewards do not repeat. These are total transition
rewards, with no added step cost on goal or collision entry.

The reward validates canonical states and feasible configuration/mode pairs,
but its numeric value does not depend on phi, the count update, theta, or the
HDR. Input histories remain unchanged. Feasible unsafe entries are obstacle
collisions and receive -100. Outward boundary commands are invalid and have no
reward transition. Staying before an obstacle is not a collision and costs -1.

## Remaining model integration

HDRs are constructed per row from a hyperstate's own count snapshot. Row-wise
credibility does not by itself establish simultaneous coverage for the complete
transition model. Belief updates, worst-case row optimization, and an augmented
POMDP wrapper for general-purpose POMDPTools simulators remain to be implemented.

The dedicated `simulate_adaptive` runner now composes an explicitly supplied
augmented kernel, transition reward, camera observation, and supplied belief
updater. It stops at goal, collision, or H executed actions, retaining the final
count update and reward. `termination_reason` gives physical events precedence
on action H, while the horizon remains external to phi. The visualization uses
an explicit HDR-checked reliable-motion row and a scripted route, without
claiming Bayesian belief learning or worst-case optimization. Its history
contains hidden states for evaluation, not as policy/updater inputs.

The final adaptive robust POMDP must use AdaptiveState = (r,m,phi) as its actual
state type, with belief updates and planning operating on hidden hyperstates.
The current DSState POMDP wiring remains temporary until those interfaces and
the augmented-state interfaces are implemented.
