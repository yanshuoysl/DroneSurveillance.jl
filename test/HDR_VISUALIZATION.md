# Supplied-parameter Dirichlet HDR test

From the repository root, run:

```bash
julia --project=test --startup-file=no test/test_hdr_visualization.jl
```

Edit the parameter list near the top of `test/test_hdr_visualization.jl`:

```julia
const DIRICHLET_INPUT = [2.0, 3.0, 2.0, 1.0, 1.0, 2.0]
```

The list itself supplies phi, and its length determines the dimension. There is
no drone environment, state, mode, action, successor enumeration, simulated count
input, or count update. The original HDR estimator is reused without changing
production simulator files. Loading the adaptive module only makes its existing
Dirichlet/HDR functions available; no POMDP environment is instantiated.

## Parameters and settings

- Supply at least three finite real parameters, each >= 1. Noninteger values such
  as `1.5` are accepted and preserved. These are Dirichlet parameters, not
  normalized probabilities. The current projection method does not support
  parameters below one, although such values are valid in a general Dirichlet.
- `K = 10_000`, `EPSILON = 0.05`, and `SEED = 42` control the original PDF-based
  Monte Carlo threshold. The code draws theta from Dir(phi), evaluates the PDF,
  and calls `quantile(densities, EPSILON)` with the existing interpolation rule.
- `VALIDATION_SAMPLES = 10_000` specifies a separate sample batch, using seed
  `SEED + 1`. `PLOT_SAMPLES = 450` selects the first samples from that batch for
  drawing; the reported coverage uses the entire validation batch.
- `GRID_RESOLUTION = 160` controls boundary detail. The existing ray/bisection or
  contour-grid method is retained, depending on the group exponents.
- With exactly three parameters and no group override, the plot uses the three
  probability components directly, in their original order. No aggregation is
  necessary, and displayed HDR membership matches the original membership.
- For longer vectors, `GROUPING = :density` keeps parameter-one entries in group
  3 and splits parameters above one into the first two groups by exponent total
  `phi[i]-1`. The rule uses only the input parameters, not sample labels. With
  only one parameter above one it occupies group 1, and the parameter-one entries
  are split between the remaining groups. With no parameter-one entries or a
  flat prior, the existing balanced partition is the fallback.
- `GROUPING = :balanced` restores the equal-parameter-total strategy.
- `GROUPS = nothing` uses those defaults. Supply three nonempty index lists,
  such as `[[1,2], [3,6], [4,5]]`, to choose your own grouping. The lists must
  partition the parameter indices exactly once. This also permits reordering
  the three components when the supplied dimension is three.

For example, `[2.0, 3.0, 2.0]` produces a direct three-component HDR.
`[1.5, 2.25, 3.0, 1.0]` tests fractional parameters and three-group projection.
A list of all ones produces constant original density, so the density-threshold
region includes the entire simplex, with 100% coverage rather than exactly 95%.
The existing estimator reports an error if sampled raw PDF values overflow or
underflow; this revision does not change that numerical policy or use log PDFs.

## Outputs

Each run regenerates files under `test/output/hdr/`:

- `hdr_simplex.png` and `hdr_simplex.svg`: parameter summary, simplex samples,
  green uncertainty region, and black boundary.
- `hdr_density.png` and `hdr_density.svg`: the same visible samples, displayed by
  their original density score `f/(f+tau)`.
- `samples.csv`: displayed coordinates, original membership, projected membership,
  raw density, display score, and all original theta components.
- `report.txt`: the supplied vector, dimension, group index mapping, threshold,
  validation coverage, and grouping comparison where applicable.

Generated files are ignored by Git. The standalone test is separate from the
normal unit-test runner because it renders figures and requires the test
project's graphics dependencies. Existing saved copies in other folders are not
modified by this script.

## Interpreting the figures

The original distribution and threshold always define the uncertainty set:

    U = {theta in the original simplex : f(theta; phi) >= tau}.

Dark points satisfy this original HDR criterion; orange rings do not. The green
area is its three-group projection when the vector has more than three entries.
It includes a grouped point if at least one compatible original vector lies in
U. Its interior boundary is `M(q) = tau`, with admitted simplex-edge pieces also
part of the boundary. No grouped Dirichlet PDF or second threshold is computed.

Projection can hide differences between original vectors, so an orange point
may lie inside the green shading. A grouping change can reduce overlap but
cannot guarantee separation. The report compares balanced and density-based
strategies on exactly the same independent validation samples. For a
three-component input, singleton groups preserve all coordinates and the
comparison reports the direct view instead.

The separate density diagnostic uses a bounded display coordinate `f/(f+tau)`:
green points have score >= 0.5 and orange points have score below 0.5. This score
is not a probability or a simplex coordinate, and changes neither the original
PDF nor the estimated threshold. It shows the original membership without
losing the density information through grouping.
