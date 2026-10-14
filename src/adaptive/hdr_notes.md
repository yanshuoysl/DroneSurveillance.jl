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

## Remaining model integration

The transition model must supply the complete, consistently ordered feasible
successor set S(r,m,a). HDRs are constructed per row from a hyperstate's own count
snapshot. Row-wise credibility does not by itself establish simultaneous coverage
for the complete transition model.

The final adaptive robust POMDP must use AdaptiveState = (r,m,phi) as its actual
state type, with belief updates and planning operating on hidden hyperstates.
The current DSState POMDP wiring remains temporary until those interfaces and
the learned transition model are implemented.
