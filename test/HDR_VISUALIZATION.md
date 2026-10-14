# Given-step manual Dirichlet HDR test

From the repository root, run:

```bash
julia --project=test test/test_hdr_visualization.jl
```

The test takes one transition row at a given robot state,
computes its original full-dimensional HDR, and plots one ternary projection.
It executes no robot transition, count increment, or full simulation. There is
no prior/after-update comparison. It writes
`test/output/hdr/hdr_simplex.png`, `hdr_simplex.svg`, `samples.csv`, and `report.txt`.
Generated files are ignored by Git and regenerated on each run. The standalone
test is intentionally separate from the normal unit-test runner because it
renders figures and uses the test environment's graphics dependencies.

## Settings

Edit the constants at the top of `test_hdr_visualization.jl`:

- `K = 10_000`, `EPSILON = 0.05`, and `SEED` control threshold estimation.
- `COUNT_SEED = 42` and `N_TWOS = 10` control the synthetic phi input. The
  example starts with ones and randomly selects ten distinct entries to set to
  two. These twos simulate a supplied count snapshot, not executed transitions.
- `ROBOT_POSITION`, `HEADING`, `MODE`, and `ACTION` select the source row.
- `GRID_SIZE`, start/goal positions, and `OBSTACLES` define the map.
- `DIRICHLET_INPUT = nothing` generates counts with the settings above. For
  manual input, replace `nothing` with the actual vector of integer counts >= 1,
  one per destination. These are neither probabilities nor increments. The
  projection formula also requires parameters >= 1.
- `SUCCESSOR_INPUT = nothing` enumerates the current tile, north, east, south,
  and west neighbors, each with headings 0, 90, 180, and 270, each paired with
  every feasible successor mode. At nominal mode this gives 5 × 4 × 7 = 140
  entries. The default source (3,2) has four free neighbors in the test map.
  This explanatory support requires five free tiles; it rejects boundary or
  obstacle neighbors rather than silently changing the dimension. It does not
  implement the production transition model. To supply another support, set
  `SUCCESSOR_INPUT` to a list of `(position=(x,y), heading=..., mode=(...))`
  entries. The count vector and successor list must have equal lengths.
- `GROUPS = nothing` greedily balances the supplied count totals across three
  groups. Set three explicit index lists to choose your own partition. At least
  three distinct successor entries are needed for a triangular visualization.
- `GRID_RESOLUTION` controls contour detail; `PLOT_SAMPLES` controls visible dots.

To test a different interior location or an existing fault, change
`ROBOT_POSITION`, `HEADING`, and `MODE`. The generated support follows those
settings; with k existing faults its dimension is 20 × (7-k). For other support
definitions, supply matching destination entries and counts explicitly.
Destination modes must preserve active faults and add at most one new fault;
their configurations must be valid free cells or the unsafe sentinel position
`(-1,-1)` with heading 0. Repeated modes with different configurations are allowed.

The default input contains 130 ones and ten twos, totaling 150 counts. Automatic
grouping gives three groups with total count 50 each. `report.txt` records all
140 destinations, their count values, and their group memberships. The figure
shows the first ten destination labels to keep the plot compact. With so few
twos in a high-dimensional row, the projection can cover almost the whole
triangle even though the original HDR has approximately 95% posterior mass.
For a flat full Dirichlet distribution, set `N_TWOS = 0` with generated input.
The green region then fills the triangle.

## What is being tested

The test uses the real adaptive map and hyperstate/count/HDR helpers. It stores
each supplied count as `phi[key] - 1` in the existing sparse representation, so
reading the row returns the input vector exactly. This storage conversion is
not a count update. Other, unprovided rows retain their implicit all-one counts.
No action is executed and no fault timing or locomotion model is assumed.
Production simulator source files are unchanged.

The original row's PDF values and `quantile` determine its threshold. No grouped
Dirichlet PDF or second threshold is used. For each triangle point, the helper
maximizes the original PDF over all compatible within-group allocations. Green
means that at least one allocation is in the original HDR. The interior black
line traces `M(q) = tau`. When every group has a positive total exponent
`sum(phi[i]-1)`, the projected region is convex with an interior mode. The helper
traces rays from that mode toward the triangle perimeter and bisects each ray
using the original PDF threshold. This resolves boundaries close to the edges
without coarse density interpolation. Other cases use the existing contour grid;
admitted simplex-edge portions are also drawn black. `GRID_RESOLUTION` controls
the number of perimeter samples or the fallback grid resolution.

Dark dots belong to the original HDR; orange rings do not. An orange ring can
project inside green because a different hidden allocation at the same grouped
coordinates may belong to the HDR. The plot reports original and projected
held-out coverage separately. The initial all-one full prior has constant density
and its projected uncertainty set is the entire triangle, with 100% coverage.
