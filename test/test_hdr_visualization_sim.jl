# Run from the repository root:
# julia --project=test test/test_hdr_visualization.jl
# Standalone given-step snapshot; no transition or count update is executed.
module HDRVisualizationTest

using Random

# ---- Editable experiment settings -------------------------------------------
const K = 10_000
const EPSILON = 0.05
const SEED = 42
const COUNT_SEED = 42
# Simulate phi for this given-step example: randomly choose N_TWOS entries
# to have count 2; all other entries have count 1. No transitions are simulated.
const N_TWOS = 10
const GRID_RESOLUTION = 160
const VALIDATION_SAMPLES = 10_000
const PLOT_SAMPLES = 450
const GRID_SIZE = (5, 5)
const START_POSITION = (1, 1)
const GOAL_POSITION = (5, 5)
const ROBOT_POSITION = (3, 2)
const HEADING = 0
const MODE = (false, false, false, false, false, false)
const ACTION = :forward
const OBSTACLES = ((2, 3), (4, 4))
# nothing generates simulated counts using COUNT_SEED and N_TWOS above.
# Or supply actual integer counts >= 1 for ONE row (r,m,a), in successor order.
const DIRICHLET_INPUT = nothing
# nothing enumerates current/north/east/south/west tiles, each with headings
# 0/90/180/270 and all feasible successor modes.
# This example requires five free tiles; it does not define production dynamics.
# Or supply [(position=(...), heading=..., mode=(...)), ...] explicitly.
const SUCCESSOR_INPUT = nothing
# :density separates counts > 1 from counts == 1, instead of equalizing totals.
# :balanced restores the previous grouping. Neither guarantees class separation.
const GROUPING = :density
# Set three explicit index lists to override GROUPING, or leave nothing.
const GROUPS = nothing
const OUTPUT_DIR = joinpath(@__DIR__, "output", "hdr")
# ----------------------------------------------------------------------------

push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))
using Test
using Compose
using Colors
import Contour
import Cairo, Fontconfig

include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
using .DroneSurveillanceAdaptive
using .DroneSurveillanceAdaptive: pdf
const DS = DroneSurveillanceAdaptive

include("hdr_projection_helpers.jl")

"""
Group by the exponents in f(theta;phi) ∝ product(theta[i]^(phi[i]-1)).
Keep count-one entries together, since their exponents are zero; split the
remaining entries between two groups by exponent total. This uses only counts,
never the validation labels, and does not change the original HDR membership.
"""
function density_groups(counts)
    length(counts) >= 3 || throw(ArgumentError("at least three successors are needed"))
    all(a -> isfinite(a) && a >= 1, counts) ||
        throw(ArgumentError("grouping requires finite counts >= 1"))
    informative = sort(findall(>(1), counts); by=i -> (1-counts[i], i))
    neutral = findall(==(1), counts)
    if length(informative) >= 2 && !isempty(neutral)
        groups = [Int[], Int[], neutral]
        weights = zeros(2)
        for i in informative
            g = argmin(weights)
            push!(groups[g], i)
            weights[g] += counts[i]-1
        end
        foreach(sort!, groups)
        return validate_groups(groups, length(counts))
    elseif length(informative) == 1 && length(neutral) >= 2
        return validate_groups([informative, neutral[1:2:end], neutral[2:2:end]], length(counts))
    end
    # With no count-one entries, or a flat prior, use the existing partition.
    return balanced_groups(counts)
end

"""Bounded display of original density: f/(f+tau), with HDR cutoff at 1/2.
This is a plotting coordinate, not a probability or a replacement HDR model.
"""
function density_score(value, tau)
    if value >= tau
        return inv(1 + tau/value)
    end
    ratio = value/tau
    return ratio/(1+ratio)
end

"""Example Cartesian support: five free tiles × four headings × successor modes."""
function neighborhood_successors(pomdp, config, mode)
    x, y = config.quad
    positions = [(x, y), (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    all(p -> 1 <= p[1] <= pomdp.size[1] && 1 <= p[2] <= pomdp.size[2] &&
             !(DSPos(p) in pomdp.obstacles), positions) ||
        throw(ArgumentError("the five-tile example requires an interior source with four free neighbors; choose another position or supply SUCCESSOR_INPUT"))
    return [(DSState(position, heading), next_mode)
            for position in positions for heading in (0, 90, 180, 270)
            for next_mode in successor_modes(mode)]
end

"""Generate a synthetic phi snapshot; never execute transitions or update counts."""
function simulated_counts(dimension; n_twos=N_TWOS, seed=COUNT_SEED)
    0 <= n_twos <= dimension || throw(ArgumentError("N_TWOS must be between 0 and the row dimension"))
    counts = ones(Int, dimension)
    selected = randperm(MersenneTwister(seed), dimension)[1:n_twos]
    counts[selected] .= 2
    return counts
end

"""Store supplied row counts directly in a hyperstate, without updating them."""
function manual_snapshot(pomdp, config, mode, action, successors, dirichlet_input)
    length(dirichlet_input) == length(successors) ||
        throw(ArgumentError("DIRICHLET_INPUT must have one count per SUCCESSOR_INPUT entry"))
    length(dirichlet_input) >= 3 ||
        throw(ArgumentError("a ternary visualization needs at least three successor entries"))
    all(a -> a isa Real && isfinite(a) && a >= 1 && isinteger(a), dirichlet_input) ||
        throw(ArgumentError("DIRICHLET_INPUT must contain integer counts >= 1"))
    source = adaptive_state(pomdp, config, mode)
    is_terminal(source) && throw(ArgumentError("choose a nonterminal source configuration"))
    keys = [CountKey(source.config, source.mode, action, rp, mp) for (rp, mp) in successors]
    length(unique(keys)) == length(keys) || throw(ArgumentError("successor entries must be distinct"))
    # DirichletCounts stores phi-1 internally; this conversion does not simulate
    # observations. count_value(phi,key) is exactly the supplied phi entry.
    counts = DirichletCounts(Dict(key => Int(a) - 1 for (key, a) in zip(keys, dirichlet_input)))
    return adaptive_state(pomdp, source.config, source.mode, counts)
end

function make_experiment(; dirichlet_input=DIRICHLET_INPUT, successor_input=SUCCESSOR_INPUT,
                         groups_input=GROUPS, grouping=GROUPING)
    pomdp = DroneSurveillancePOMDP(size=GRID_SIZE, region_A=DSPos(START_POSITION),
        region_B=DSPos(GOAL_POSITION), obstacles=DSPos[DSPos(p) for p in OBSTACLES])
    action = ACTIONS_DICT[ACTION]
    config = DSState(ROBOT_POSITION, HEADING)
    successors = isnothing(successor_input) ? neighborhood_successors(pomdp, config, MODE) :
        Tuple{DSState, FaultMode}[(DSState(item.position, item.heading), item.mode)
                                 for item in successor_input]
    input_counts = isnothing(dirichlet_input) ? simulated_counts(length(successors)) : dirichlet_input
    state = manual_snapshot(pomdp, DSState(ROBOT_POSITION, HEADING), MODE,
                            action, successors, input_counts)
    stored_counts = copy(state.counts.increments)
    hdr = build_row_hdr(pomdp, state, action, successors;
        epsilon=EPSILON, n_samples=K, rng=MersenneTwister(SEED))
    groups = if !isnothing(groups_input)
        validate_groups(deepcopy(groups_input), length(successors))
    elseif grouping == :density
        density_groups(hdr.distribution.alpha)
    elseif grouping == :balanced
        balanced_groups(hdr.distribution.alpha)
    else
        throw(ArgumentError("GROUPING must be :density or :balanced"))
    end
    selected_grouping = isnothing(groups_input) ? grouping : :manual
    return (; pomdp, state, action, successors, groups, hdr, stored_counts,
             grouping=selected_grouping,
             input_counts=Int.(input_counts))
end

function validation_samples(hdr, groups, seed)
    samples = rand(MersenneTwister(seed), hdr.distribution, VALIDATION_SAMPLES)
    inside_full = [contains(hdr, theta) for theta in eachcol(samples)]
    grouped = [group_probabilities(theta, groups) for theta in eachcol(samples)]
    inside_projection = [projected_contains(hdr, groups, q) for q in grouped]
    densities = [pdf(hdr.distribution, theta) for theta in eachcol(samples)]
    scores = density_score.(densities, hdr.tau)
    return (; samples, grouped, inside_full, inside_projection, densities, scores,
        coverage=sum(inside_full) / VALIDATION_SAMPLES,
        projected_coverage=sum(inside_projection) / VALIDATION_SAMPLES)
end

"""Compare projections of exactly the same independent validation samples."""
function grouping_comparison(e, samples)
    candidates = ((:balanced, balanced_groups(e.input_counts)),
                  (:density, density_groups(e.input_counts)))
    return [let
        included = [projected_contains(e.hdr, groups, group_probabilities(theta, groups))
                    for theta in eachcol(samples.samples)]
        outside = .!samples.inside_full
        (; strategy, totals=[sum(e.input_counts[g]) for g in groups],
           orange_total=sum(outside), orange_outside=sum(outside .& .!included),
           projected_coverage=sum(included)/length(included))
    end for (strategy, groups) in candidates]
end

function run_checks(experiment, samples, curves)
    e = experiment
    @testset "Given-step manual Dirichlet HDR" begin
        # The default nominal example includes every configuration-mode pair.
        model = DroneSurveillancePOMDP(obstacles=DSPos[])
        support = neighborhood_successors(model, DSState([3, 3], 0), NOMINAL_MODE)
        @test length(support) == length(unique(support)) == 140
        @test Set(config.quad for (config, _) in support) ==
              Set(DSPos.([(3, 3), (3, 4), (4, 3), (3, 2), (2, 3)]))
        @test Set(config.heading for (config, _) in support) == Set((0, 90, 180, 270))
        @test all(count(pair -> pair[2] == mode, support) == 20
                  for mode in successor_modes(NOMINAL_MODE))
        phi = simulated_counts(140; n_twos=10, seed=42)
        @test count(==(1), phi) == 130
        @test count(==(2), phi) == 10
        @test phi == simulated_counts(140; n_twos=10, seed=42)
        informative_groups = density_groups(phi)
        @test [sum(phi[g]) for g in informative_groups] == [10,10,130]
        @test all(phi[i] == 2 for g in informative_groups[1:2] for i in g)
        @test Set(informative_groups[3]) == Set(findall(==(1),phi))
        @test density_groups(ones(6)) == balanced_groups(ones(6))
        @test density_groups([2,1,1,1]) == [[1],[2,4],[3]]
        @test density_groups([2,2,1]) == [[1],[2],[3]]
        @test_throws ArgumentError density_groups([2,0,1])
        @test_throws ArgumentError density_groups([2,1])
        @test density_score(0.0,2.0) == 0.0
        @test density_score(2.0,2.0) == 0.5
        @test density_score(1e300,1e300) == 0.5
        @test_throws ArgumentError simulated_counts(140; n_twos=141)
        @test_throws ArgumentError neighborhood_successors(model, DSState([1, 1], 0), NOMINAL_MODE)
        @test e.state.config == DSState(ROBOT_POSITION, HEADING)
        @test e.state.mode == MODE
        @test e.hdr.distribution.alpha == e.input_counts
        @test e.state.counts.increments == e.stored_counts
        for ((config, mode), count) in zip(e.successors, e.input_counts)
            key = CountKey(e.state.config, e.state.mode, e.action, config, mode)
            @test count_value(e.state.counts, key) == count
        end
        @test sort(vcat(e.groups...)) == collect(eachindex(e.successors))
        @test all(group -> !isempty(group), e.groups)
        @test e.hdr.successors == e.successors
        @test all(isfinite, samples.densities)
        @test all(s -> 0 <= s <= 1, samples.scores)
        @test (samples.scores .>= 0.5) == samples.inside_full
        if all(==(1), e.hdr.distribution.alpha)
            @test samples.coverage == 1.0
            @test samples.projected_coverage == 1.0
        else
            @test abs(samples.coverage - (1 - EPSILON)) < 0.025
        end
        @test e.hdr.tau == estimate_tau(e.input_counts;
            epsilon=EPSILON, n_samples=K, rng=MersenneTwister(SEED))

        # Manually chosen state/mode/support and counts, without a motion fixture.
        alternate_model = DroneSurveillancePOMDP(obstacles=DSPos[])
        alternate_mode = (false, true, false, false, false, false)
        alternate_config = DSState([3, 3], 90)
        alternate_support = [(DSState([4, 3], 90), alternate_mode),
                             (alternate_config, alternate_mode),
                             (DSState([3, 4], 90), alternate_mode)]
        alternate = manual_snapshot(alternate_model, alternate_config, alternate_mode,
                                    1, alternate_support, [5, 3, 2])
        alternate_hdr = build_row_hdr(alternate_model, alternate, 1, alternate_support;
                                     epsilon=EPSILON, n_samples=100, rng=MersenneTwister(SEED))
        @test alternate.config == alternate_config
        @test alternate.mode == alternate_mode
        @test alternate_hdr.distribution.alpha == [5, 3, 2]
        @test length(unique(mode for (_, mode) in alternate_hdr.successors)) == 1
        @test_throws ArgumentError manual_snapshot(alternate_model, alternate_config,
            alternate_mode, 1, alternate_support, [5, 3])
        @test_throws ArgumentError manual_snapshot(alternate_model, alternate_config,
            alternate_mode, 1, alternate_support, [0, 3, 2])
        @test_throws ArgumentError manual_snapshot(alternate_model, alternate_config,
            alternate_mode, 1, alternate_support, [1.5, 3, 2])
        @test_throws ArgumentError manual_snapshot(alternate_model, alternate_config,
            alternate_mode, 1, [alternate_support[1], alternate_support[1], alternate_support[2]], [5, 3, 2])

        for i in 1:VALIDATION_SAMPLES
            q = samples.grouped[i]
            @test isapprox(sum(q), 1)
            @test !samples.inside_full[i] || samples.inside_projection[i]
        end
        for i in 1:min(100, VALIDATION_SAMPLES)
            value = profile_pdf(e.hdr, e.groups, samples.grouped[i])
            original = pdf(e.hdr.distribution, samples.samples[:, i])
            @test value >= original || isapprox(value, original; rtol=1e-12)
        end

        # Independently grid-search the hidden allocation for a five-component case.
        alpha = [3.0, 5.0, 2.0, 4.0, 6.0]
        groups = [[1], [2], [3, 4, 5]]
        q = [0.25, 0.25, 0.5]
        distribution = DS.Dirichlet(alpha)
        allocation = maximizing_allocation(alpha, groups, q)
        analytic = pdf(distribution, allocation)
        brute_force = maximum(pdf(distribution,
            [q[1], q[2], 0.5i / 100, 0.5j / 100, 0.5(100 - i - j) / 100])
            for i in 0:100 for j in 0:100-i)
        @test group_probabilities(allocation, groups) ≈ q
        @test analytic >= brute_force || isapprox(analytic, brute_force)
        @test isapprox(analytic, brute_force; rtol=0.005)
        @test maximizing_allocation(ones(5), groups, q)[3:5] ≈ fill(1 / 6, 3)
        @test_throws ArgumentError validate_groups([[1], [2], [2, 3]], 3)
        @test_throws ArgumentError validate_groups([[1], [2], Int[]], 2)
        @test_throws ArgumentError maximizing_allocation(alpha, groups, [0.3, 0.3, 0.3])
        @test_throws ArgumentError maximizing_allocation([0.5, 2, 2], [[1], [2], [3]], q)

        # Check the interior contour against the ORIGINAL PDF threshold.
        if all(sum(e.hdr.distribution.alpha[i] - 1 for i in group) > 0 for group in e.groups)
            @test !isempty(curves)
            for curve in curves, point in curve
                @test isapprox(profile_pdf(e.hdr, e.groups, collect(point)),
                               e.hdr.tau; rtol=0.04)
            end
        end
        # A separate uniform-input edge-case check, not a second figure or update.
        uniform = manual_snapshot(e.pomdp, e.state.config, e.state.mode,
                                  e.action, e.successors, ones(Int, length(e.successors)))
        uniform_hdr = build_row_hdr(e.pomdp, uniform, e.action, e.successors;
            epsilon=EPSILON, n_samples=100, rng=MersenneTwister(SEED))
        @test isempty(uniform.counts.increments)
        @test isempty(projected_contours(uniform_hdr, e.groups, 20))
        @test all(projected_contains(uniform_hdr, e.groups, q)
                  for q in ([1.0,0,0], [0.0,1,0], [0.0,0,1], [1/3,1/3,1/3]))
        @test e.state.counts.increments == e.stored_counts
    end
end

# Compose draws the figure; the geometry helper traces M(q)=tau.
const FIGURE_WIDTH = 1200
const FIGURE_HEIGHT = 1150
figure_height(e) = max(FIGURE_HEIGHT, 1090 + 23cld(min(length(e.successors), 10), 2))
const GREEN = "#cbeecf"
const INK = "#19392c"
const ORANGE = "#d46b22"

label(x, y, value; size=11, color="#283b36", align=hleft) =
    compose(context(), Compose.text(x, y, value, align, vcenter),
            fontsize(size * pt), fill(color), font("DejaVu Sans"))
poly(points; color=GREEN, edge=nothing, width=0.5) =
    compose(context(), polygon(points), fill(color), stroke(edge), linewidth(width * px))
path(points; color=INK, width=1.0) =
    compose(context(), line(points), fill(nothing), stroke(color), linewidth(width * px))
dot(x, y; color=INK, radius=2.4, hollow=false) =
    compose(context(), circle(x, y, radius), fill(hollow ? nothing : color),
            stroke(color), linewidth(1.0px))

function assemble_figure(elements, width, height)
    # The count-one grouping can shade thousands of small grid triangles.
    # Compose's variadic interface recurses per child, so attach iteratively.
    figure = context(units=UnitBox(0,0,width,height))
    for element in Iterators.reverse(elements)
        compose!(figure,element)
    end
    return figure
end

function add_environment!(elements, e)
    push!(elements, label(45, 156, "Adaptive drone environment"; size=14))
    left, top, side = 45.0, 200.0, 300.0
    nx, ny = e.pomdp.size
    cellw, cellh = side / nx, side / ny
    for x in 1:nx, y in 1:ny
        pos = DSPos(x, y)
        color = pos in e.pomdp.obstacles ? "#e8a149" :
            pos == e.pomdp.region_A || pos == e.pomdp.region_B ? "#d9eaca" : "#f4f7f5"
        x0, y0 = left + (x - 1) * cellw, top + (ny - y) * cellh
        push!(elements, poly([(x0,y0), (x0+cellw,y0), (x0+cellw,y0+cellh), (x0,y0+cellh)];
                             color, edge="#acbab2", width=1))
        if pos == e.pomdp.region_A || pos == e.pomdp.region_B
            push!(elements, label(x0 + cellw/2, y0 + cellh/2,
                pos == e.pomdp.region_A ? "A" : "B"; size=15, align=hcenter))
        end
    end
    x, y = e.state.config.quad
    cx, cy = left + (x - 0.5) * cellw, top + (ny - y + 0.5) * cellh
    push!(elements, dot(cx, cy; color="#1f6276", radius=min(cellw, cellh) * 0.17))
    dx, dy = DS.HEADING_DIRS[div(e.state.config.heading, 90) + 1]
    length_arrow = min(cellw, cellh) * 0.34
    tip = (cx + dx * length_arrow, cy - dy * length_arrow)
    push!(elements, path([(cx, cy), tip]; color="#153c4b", width=3))
    push!(elements, dot(tip...; color="#153c4b", radius=3))
    push!(elements, label(45, 535, "r = ($(x), $(y), $(e.state.config.heading) deg)"; size=12))
    push!(elements, label(45, 563, "m = " * join(Int.(e.state.mode)); size=12))
    push!(elements, label(45, 591, "action = $(ACTION)"; size=12))
    push!(elements, label(45, 630, "GIVEN-STEP SNAPSHOT"; size=11, color="#92501e"))
    support_labels = isnothing(SUCCESSOR_INPUT) ?
        ("Current tile + four neighbors.", "Four headings; feasible next modes.") :
        ("Manually supplied successor support.", "Counts read directly from input.")
    for (i, text) in enumerate((support_labels..., "No transition or count update."))
        push!(elements, label(45, 653 + 23(i-1), text; size=10))
    end
end

function add_simplex!(elements, hdr, groups, samples, curves, left, title)
    baseline, side = 670.0, 485.0
    xy(q) = ternary_xy(q, left, baseline, side)
    vertices = xy.(((1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)))
    center = left + side / 2
    push!(elements, label(center, 156, title; size=15, align=hcenter))
    count_string = join(Int.(hdr.distribution.alpha[1:min(8, end)]), ", ")
    length(hdr.distribution.alpha) > 8 && (count_string *= ", ...; see report")
    push!(elements, label(center, 187, "Full counts: [$(count_string)]"; size=10, align=hcenter))
    push!(elements, poly(collect(vertices); color="#f6faf7", edge="#9eafa3", width=1.2))
    for polygon in projected_fill_polygons(hdr, groups, GRID_RESOLUTION)
        push!(elements, poly(xy.(polygon)))
    end
    for t in 0.2:0.2:0.8
        for g in 1:3
            other = filter(!=(g), collect(1:3))
            a, b = zeros(3), zeros(3)
            a[g] = b[g] = t
            a[other[1]] = b[other[2]] = 1-t
            push!(elements, path([xy(a), xy(b)]; color="#b5c9b9", width=0.65))
        end
        push!(elements, label(left + side*t, baseline + 19, string(round(t; digits=1));
                             size=8, align=hcenter))
        q1tick = [t, 0.0, 1-t]
        q3tick = [0.0, 1-t, t]
        x1,y1 = xy(q1tick)
        x3,y3 = xy(q3tick)
        push!(elements, label(x1-12, y1, string(round(t; digits=1)); size=8, align=hright))
        push!(elements, label(x3+12, y3, string(round(t; digits=1)); size=8))
    end
    for i in 1:min(PLOT_SAMPLES, length(samples.grouped))
        x, y = xy(samples.grouped[i])
        inside = samples.inside_full[i]
        push!(elements, dot(x, y; color=inside ? INK : ORANGE,
                            radius=inside ? 2.3 : 3.8, hollow=!inside))
    end
    for curve in curves
        push!(elements, path(xy.(curve); color="#0d1e15", width=2.4))
    end
    # Boundary portions on the simplex edges are part of the projected region.
    for (a,b) in (([1.0,0,0],[0.0,1,0]), ([0.0,1,0],[0.0,0,1]), ([0.0,0,1],[1.0,0,0]))
        for i in 0:GRID_RESOLUTION-1
            t1, t2 = i / GRID_RESOLUTION, (i+1) / GRID_RESOLUTION
            q1, q2 = (1-t1).*a .+ t1.*b, (1-t2).*a .+ t2.*b
            if projected_contains(hdr, groups, (q1.+q2)./2)
                push!(elements, path([xy(q1), xy(q2)]; color="#0d1e15", width=2.0))
            end
        end
    end
    push!(elements, label(left, baseline + 50, "q1 = 1"; size=12, align=hcenter))
    push!(elements, label(left + side, baseline + 50, "q2 = 1"; size=12, align=hcenter))
    push!(elements, label(center, vertices[3][2] - 22, "q3 = 1"; size=12, align=hcenter))
    push!(elements, label(center, 761, "Original tau = $(round(hdr.tau; sigdigits=5))";
                         size=10, align=hcenter))
    push!(elements, label(center, 787,
        "Full HDR coverage: $(round(100samples.coverage; digits=2))%"; size=10, align=hcenter))
    push!(elements, label(center, 813,
        "Projected coverage: $(round(100samples.projected_coverage; digits=2))%"; size=10, align=hcenter))
    if all(==(1), hdr.distribution.alpha)
        push!(elements, label(center, 839, "Constant density: the full triangle is included";
                             size=9, align=hcenter))
    end
end

function make_figure(e, samples, curves)
    height = figure_height(e)
    elements = Any[poly([(0,0),(FIGURE_WIDTH,0),(FIGURE_WIDTH,height),(0,height)]; color="white")]
    push!(elements, label(45, 44, "Given-step Dirichlet input: projected 95% HDR"; size=22))
    push!(elements, label(45, 82,
        "Original $(length(e.successors))-component PDF and threshold | epsilon = $(EPSILON) | K = $(K) | seed = $(SEED)"; size=12))
    add_environment!(elements, e)
    add_simplex!(elements, e.hdr, e.groups, samples, curves, 560.0,
        "Grouping: $(e.grouping)")
    push!(elements, label(45, 738,
        "Counts: $(count(==(1), e.input_counts)) ones, $(count(==(2), e.input_counts)) twos"; size=11))
    push!(elements, label(45, 764,
        "Group totals: " * join([sum(e.input_counts[g]) for g in e.groups], ", "); size=11))
    if e.grouping == :density && all(e.input_counts[i] > 1 for g in e.groups[1:2] for i in g) &&
       all(e.input_counts[i] == 1 for i in e.groups[3])
        push!(elements, label(45, 791, "q1, q2: split counts above one"; size=10))
        push!(elements, label(45, 815, "q3: count-one entries"; size=10))
    end

    push!(elements, poly([(45,881),(65,881),(65,897),(45,897)]; color=GREEN, edge=INK))
    push!(elements, label(76, 890, "Projected uncertainty region"; size=10))
    push!(elements, path([(440,890),(471,890)]; width=2.4))
    push!(elements, label(482, 890, "M(q) = original tau"; size=10))
    push!(elements, dot(53,923))
    push!(elements, label(68, 923, "Inside original HDR"; size=10))
    push!(elements, dot(447,923; color=ORANGE, radius=3.8, hollow=true))
    push!(elements, label(462, 923, "Outside original HDR"; size=10))
    push!(elements, label(45, 964,
        "Grouping hides allocations: an orange point may lie inside green."; size=11))
    group_text = join(["q$(g) = " * join(indices, "+") for (g,indices) in enumerate(e.groups)], "     |     ")
    length(group_text) > 100 && (group_text = "See report.txt for all group memberships")
    push!(elements, label(45, 995, "Groups (successor IDs): " * group_text; size=11))
    push!(elements, label(45, 1025,
        "First $(min(10, length(e.successors))) of $(length(e.successors)) successors; complete mapping and counts in report.txt"; size=10))
    for (i, (config, mode)) in enumerate(Iterators.take(e.successors, 10))
        row, column = divrem(i-1, 2)
        config_text = config == UNSAFE_CONFIG ? "unsafe terminal" :
            "r'=($(config.quad[1]),$(config.quad[2]),$(config.heading))"
        push!(elements, label(45 + 590column, 1055 + 23row,
            "$(i): $(config_text), m'=$(join(Int.(mode)))"; size=10))
    end
    # Compose paints earlier children on top; reverse our painter-order list.
    return assemble_figure(elements, FIGURE_WIDTH, height)
end

"""Show original HDR membership without discarding the density information."""
function make_density_figure(e, samples)
    width, height = 1100, 650
    left, right, top, bottom = 90.0, 1040.0, 160.0, 490.0
    count = min(PLOT_SAMPLES, length(samples.scores))
    x(i) = left + (right-left)*(i-1)/max(1,count-1)
    y(score) = bottom - (bottom-top)*score
    elements = Any[poly([(0,0),(width,0),(width,height),(0,height)]; color="white")]
    push!(elements, label(45,44,"Original HDR membership: density diagnostic"; size=21))
    push!(elements, label(45,80,
        "Same $(count) samples as simplex | original $(length(e.input_counts))-component PDF | epsilon = $(EPSILON)"; size=12))
    push!(elements, label(left,125,"Display score = f(theta; phi) / (f(theta; phi) + tau)"; size=12))
    push!(elements, poly([(left,top),(right,top),(right,y(0.5)),(left,y(0.5))]; color="#edf7ee"))
    push!(elements, poly([(left,y(0.5)),(right,y(0.5)),(right,bottom),(left,bottom)]; color="#fff5ec"))
    for score in (0.0,0.25,0.5,0.75,1.0)
        push!(elements,path([(left,y(score)),(right,y(score))]; color="#d0d8d3",width=0.7))
        push!(elements,label(left-12,y(score),string(score); size=10,align=hright))
    end
    push!(elements,path([(left,top),(left,bottom),(right,bottom)]; width=1.2))
    push!(elements,path([(left,y(0.5)),(right,y(0.5))]; width=1.8))
    for i in 1:count
        inside = samples.inside_full[i]
        push!(elements,dot(x(i),y(samples.scores[i]);color=inside ? INK : ORANGE,
                            radius=inside ? 2.5 : 3.8,hollow=!inside))
    end
    for i in unique(round.(Int,range(1,max(1,count);length=4)))
        push!(elements,label(x(i),bottom+20,string(i);size=10,align=hcenter))
    end
    push!(elements,label((left+right)/2,bottom+51,"Sample index (not simplex coordinates)";size=12,align=hcenter))
    push!(elements,label(45,580,"Cutoff 0.5 corresponds exactly to f(theta; phi) = tau.";size=12))
    push!(elements,label(45,608,"Green: score >= 0.5. Orange: score below 0.5. The score is not a probability.";size=11))
    return assemble_figure(elements,width,height)
end

function write_report(e, samples, comparison)
    open(joinpath(OUTPUT_DIR, "report.txt"), "w") do io
        println(io, "Given-step manual Dirichlet input: projection of the original full-dimensional 95% HDR")
        println(io, "Settings: K=$K, epsilon=$EPSILON, seed=$SEED, resolution=$GRID_RESOLUTION")
        println(io, "Count input: ", isnothing(DIRICHLET_INPUT) ? "simulated phi snapshot; COUNT_SEED=$COUNT_SEED, N_TWOS=$N_TWOS" : "manually supplied phi snapshot")
        println(io, "Source: r=$(e.state.config), m=$(e.state.mode), action=$ACTION")
        println(io, "Input counts: $(e.input_counts)")
        println(io, "Support: ", isnothing(SUCCESSOR_INPUT) ? "current tile + four neighbors, four headings, feasible successor modes" : "explicit manually supplied support")
        println(io, "No transition, count update, or episode is simulated.")
        println(io, "Selected grouping strategy: $(e.grouping)")
        println(io, "Group count totals: $([sum(e.input_counts[g]) for g in e.groups])")
        println(io, "Groups: $(e.groups)")
        for (i, (config, mode)) in enumerate(e.successors)
            println(io, "Successor $i: r'=$config, m'=$mode, count=$(e.input_counts[i])")
        end
        println(io, "Original tau=$(e.hdr.tau)")
        println(io, "Held-out original coverage=$(samples.coverage), projected coverage=$(samples.projected_coverage)")
        println(io, "Grouping comparison on the SAME independent samples (not used to choose the grouping):")
        for c in comparison
            println(io, "  $(c.strategy): totals=$(c.totals), orange outside projection=$(c.orange_outside)/$(c.orange_total), projected coverage=$(c.projected_coverage)")
        end
        println(io, "Complete green/orange separation is not guaranteed by a three-group projection.")
        println(io, "hdr_density.png plots the original density score f/(f+tau), with the SAME membership cutoff at 0.5. This display score is not a probability.")
    end
    open(joinpath(OUTPUT_DIR, "samples.csv"), "w") do io
        println(io, join(vcat(["q1", "q2", "q3", "inside_full_hdr", "inside_projection", "density", "density_score"],
                             ["theta_$i" for i in eachindex(e.successors)]), ","))
        for i in 1:min(PLOT_SAMPLES, VALIDATION_SAMPLES)
            println(io, join(vcat(samples.grouped[i],
                [samples.inside_full[i], samples.inside_projection[i], samples.densities[i], samples.scores[i]], samples.samples[:,i]), ","))
        end
    end
end

function main()
    experiment = make_experiment()
    samples = validation_samples(experiment.hdr, experiment.groups, SEED + 1)
    curves = projected_contours(experiment.hdr, experiment.groups, GRID_RESOLUTION)
    run_checks(experiment, samples, curves)
    comparison = grouping_comparison(experiment, samples)
    mkpath(OUTPUT_DIR)
    figure = make_figure(experiment, samples, curves)
    height = figure_height(experiment)
    draw(SVG(joinpath(OUTPUT_DIR, "hdr_simplex.svg"), FIGURE_WIDTH*px, height*px), figure)
    draw(PNG(joinpath(OUTPUT_DIR, "hdr_simplex.png"), FIGURE_WIDTH*px, height*px; dpi=120), figure)
    density_figure = make_density_figure(experiment,samples)
    draw(SVG(joinpath(OUTPUT_DIR,"hdr_density.svg"),1100px,650px),density_figure)
    draw(PNG(joinpath(OUTPUT_DIR,"hdr_density.png"),1100px,650px;dpi=120),density_figure)
    write_report(experiment, samples, comparison)
    println("Manual Dirichlet input: ", experiment.input_counts)
    println("Groups: ", experiment.groups)
    println("Original held-out HDR coverage: ", samples.coverage)
    println("Projected held-out coverage: ", samples.projected_coverage)
    for c in comparison
        println("$(c.strategy): orange outside projection = $(c.orange_outside)/$(c.orange_total); group totals = $(c.totals)")
    end
    println("Saved simplex and density PNG/SVG, samples.csv, and report.txt to ", OUTPUT_DIR)
    return experiment
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HDRVisualizationTest.main()
end
