# Run from the repository root:
# julia --project=test --startup-file=no test/test_hdr_visualization_.jl
# Direct three-component HDR example. No grouping, transition, or count update.
# Results: test/output/hdr/three_inputs/ (separate from the 140-entry example).
module HDRThreeInputVisualizationTest

using Random

# ---- Editable experiment settings -------------------------------------------
const K = 10_000
const EPSILON = 0.05
const SEED = 42
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
# Actual phi values for this illustrative three-outcome transition row.
# These are supplied counts, not probabilities, increments, or grouped counts.
const DIRICHLET_INPUT = [2,2,1] #[2, 3, 2]
# One destination (r',m') for each input count, in the same order.
# This reduced test support illustrates the HDR; it is not the full 140-entry row.
const SUCCESSOR_INPUT = [
    (position=ROBOT_POSITION, heading=HEADING, mode=MODE),
    (position=(3, 3), heading=HEADING, mode=MODE),
    (position=(4, 2), heading=HEADING, mode=MODE),
]
const OUTPUT_DIR = joinpath(@__DIR__, "output", "hdr", "three_inputs")
const PHI_LABELS = ("φ₁", "φ₂", "φ₃")
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

"""Store supplied row counts directly in a hyperstate, without updating them."""
function manual_snapshot(pomdp, config, mode, action, successors, dirichlet_input)
    length(dirichlet_input) == length(successors) ||
        throw(ArgumentError("DIRICHLET_INPUT must have one count per SUCCESSOR_INPUT entry"))
    length(dirichlet_input) == 3 ||
        throw(ArgumentError("this direct simplex example requires exactly three successor entries"))
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

function make_experiment(; dirichlet_input=DIRICHLET_INPUT, successor_input=SUCCESSOR_INPUT)
    pomdp = DroneSurveillancePOMDP(size=GRID_SIZE, region_A=DSPos(START_POSITION),
        region_B=DSPos(GOAL_POSITION), obstacles=DSPos[DSPos(p) for p in OBSTACLES])
    action = ACTIONS_DICT[ACTION]
    successors = Tuple{DSState, FaultMode}[(DSState(item.position, item.heading), item.mode)
                                          for item in successor_input]
    state = manual_snapshot(pomdp, DSState(ROBOT_POSITION, HEADING), MODE,
                            action, successors, dirichlet_input)
    stored_counts = copy(state.counts.increments)
    hdr = build_row_hdr(pomdp, state, action, successors;
        epsilon=EPSILON, n_samples=K, rng=MersenneTwister(SEED))
    return (; pomdp, state, action, successors, hdr, stored_counts,
             input_counts=Int.(dirichlet_input))
end

function validation_samples(hdr, seed)
    samples = rand(MersenneTwister(seed), hdr.distribution, VALIDATION_SAMPLES)
    # Each column already is (θ₁,θ₂,θ₃). No aggregation or projection occurs.
    inside = [contains(hdr, p) for p in eachcol(samples)]
    return (; samples, inside, coverage=sum(inside) / VALIDATION_SAMPLES)
end

# Component 1 at the left vertex, component 2 at the right, component 3 at the top.
# Plot positions are probabilities; the corner annotations show their input counts.
ternary_xy(p, left, baseline, side) =
    (left + side * (p[2] + p[3] / 2), baseline - side * sqrt(3) / 2 * p[3])

"""Boundary f(θ;phi)=tau, evaluated directly on the three-component simplex."""
function hdr_contours(hdr, resolution)
    resolution >= 10 || throw(ArgumentError("plot resolution must be at least 10"))
    alpha = hdr.distribution.alpha
    all(==(1), alpha) && return Vector{NTuple{3, Float64}}[]
    if all(>(1), alpha)
        # For phi_i > 1 the HDR is convex with an interior mode. Resolve its
        # boundary by bisection along rays, retaining the original PDF and tau.
        center = (alpha .- 1) ./ sum(alpha .- 1)
        vertices = ([1.0,0,0], [0.0,1,0], [0.0,0,1])
        curve = NTuple{3, Float64}[]
        for edge in 1:3, j in 0:resolution-1
            t = j / resolution
            target = (1-t) .* vertices[edge] .+ t .* vertices[mod1(edge+1, 3)]
            low, high = 0.0, 1.0
            for _ in 1:80
                middle = (low + high) / 2
                p = (1-middle) .* center .+ middle .* target
                if contains(hdr, p)
                    low = middle
                else
                    high = middle
                end
            end
            push!(curve, Tuple((1-low) .* center .+ low .* target))
        end
        push!(curve, first(curve))
        return [curve]
    end
    # For editable inputs containing ones, use the direct PDF contour grid.
    axis = collect(range(0.0, 1.0; length=resolution+1))
    field = zeros(resolution+1, resolution+1)
    for i in 0:resolution, j in 0:resolution-i
        field[i+1,j+1] = pdf(hdr.distribution,
            [i/resolution, j/resolution, (resolution-i-j)/resolution])
    end
    curves = Vector{NTuple{3, Float64}}[]
    for curve in Contour.lines(Contour.contour(axis, axis, field, hdr.tau))
        run = NTuple{3, Float64}[]
        for (x,y) in Contour.vertices(curve)
            if x >= 0 && y >= 0 && x+y <= 1+1e-12
                p = [x,y,max(0.0,1-x-y)]
                p ./= sum(p)
                push!(run, Tuple(p))
            else
                length(run) > 1 && push!(curves, run)
                run = NTuple{3, Float64}[]
            end
        end
        length(run) > 1 && push!(curves, run)
    end
    return curves
end

"""Direct PDF clipping for grid cells when an HDR touches the simplex edges."""
function hdr_fill_polygons(hdr, curves, resolution)
    all(==(1), hdr.distribution.alpha) &&
        return [[(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]]
    all(>(1), hdr.distribution.alpha) && return curves
    polygons = Vector{NTuple{3, Float64}}[]
    p(i,j) = (i/resolution, j/resolution, (resolution-i-j)/resolution)
    function add_triangle(indices)
        vertices = [p(i,j) for (i,j) in indices]
        values = [pdf(hdr.distribution, collect(point)) for point in vertices]
        polygon = NTuple{3, Float64}[]
        for i in 1:3
            j = mod1(i+1,3)
            inside_i, inside_j = values[i] >= hdr.tau, values[j] >= hdr.tau
            inside_i && push!(polygon, vertices[i])
            if inside_i != inside_j
                t = (hdr.tau-values[i]) / (values[j]-values[i])
                push!(polygon, ntuple(k -> (1-t)*vertices[i][k]+t*vertices[j][k],3))
            end
        end
        length(polygon) >= 3 && push!(polygons,polygon)
    end
    for i in 0:resolution-1, j in 0:resolution-i-1
        add_triangle(((i,j),(i+1,j),(i,j+1)))
        i+j <= resolution-2 && add_triangle(((i+1,j),(i+1,j+1),(i,j+1)))
    end
    return polygons
end

function run_checks(e, samples, curves)
    @testset "Direct three-component Dirichlet HDR" begin
        @test length(e.input_counts) == length(e.successors) == 3
        @test length(unique(e.successors)) == 3
        @test e.state.config == DSState(ROBOT_POSITION, HEADING)
        @test e.state.mode == MODE
        @test e.hdr.distribution.alpha == e.input_counts
        @test e.hdr.successors == e.successors
        for ((config,mode), count) in zip(e.successors,e.input_counts)
            key = CountKey(e.state.config,e.state.mode,e.action,config,mode)
            @test count_value(e.state.counts,key) == count
        end
        @test e.hdr.tau == estimate_tau(e.input_counts;
            epsilon=EPSILON,n_samples=K,rng=MersenneTwister(SEED))
        @test samples.samples == rand(MersenneTwister(SEED+1),e.hdr.distribution,VALIDATION_SAMPLES)
        @test all(isapprox(sum(p),1.0) for p in eachcol(samples.samples))
        for (p, inside) in zip(eachcol(samples.samples),samples.inside)
            # The same density test defines dot color and the shaded region.
            @test inside == (pdf(e.hdr.distribution,p) >= e.hdr.tau)
        end
        if all(==(1),e.input_counts)
            @test samples.coverage == 1.0
        else
            @test abs(samples.coverage-(1-EPSILON)) < 0.025
        end
        if all(>(1),e.input_counts)
            @test length(curves) == 1
            @test first(only(curves)) == last(only(curves))
            @test all(isapprox(pdf(e.hdr.distribution,collect(p)),e.hdr.tau;rtol=1e-8)
                      for p in only(curves))
        end
        # Independently known PDF: Dir(2,3,2) has f(p)=360*p1*p2^2*p3.
        distribution = DS.Dirichlet([2.0,3.0,2.0])
        @test pdf(distribution,[0.2,0.5,0.3]) ≈ 360*0.2*0.5^2*0.3
        @test ternary_xy([1,0,0],0,1,2) == (0.0,1.0)
        @test ternary_xy([0,1,0],0,1,2) == (2.0,1.0)
        @test collect(ternary_xy([0,0,1],0,1,2)) ≈ [1.0,1-sqrt(3)]
        @test_throws ArgumentError manual_snapshot(e.pomdp,e.state.config,e.state.mode,
            e.action,e.successors,[2,3])
        @test_throws ArgumentError manual_snapshot(e.pomdp,e.state.config,e.state.mode,
            e.action,e.successors,[2,0,2])
        @test_throws ArgumentError manual_snapshot(e.pomdp,e.state.config,e.state.mode,
            e.action,e.successors,[2,1.5,2])
        uniform = make_experiment(dirichlet_input=[1,1,1])
        @test isempty(hdr_contours(uniform.hdr,20))
        @test all(contains(uniform.hdr,p) for p in ([1.0,0,0],[0.0,1,0],[0.0,0,1],[1/3,1/3,1/3]))
        @test length(hdr_fill_polygons(uniform.hdr,[],20)) == 1
        @test e.state.counts.increments == e.stored_counts
    end
end

# Compose draws the direct three-component PDF region and its boundary.
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
    for (i, text) in enumerate(("Three illustrative successor pairs.",
                               "Direct phi = ($(join(e.input_counts, ", "))); no grouping.",
                               "No transition or count update."))
        push!(elements, label(45, 653 + 23(i-1), text; size=10))
    end
end

function add_simplex!(elements, hdr, samples, curves, left, title)
    baseline, side = 670.0, 485.0
    xy(p) = ternary_xy(p, left, baseline, side)
    vertices = xy.(((1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)))
    center = left + side / 2
    push!(elements, label(center, 156, title; size=15, align=hcenter))
    push!(elements, poly(collect(vertices); color="#f6faf7", edge="#9eafa3", width=1.2))
    for polygon in hdr_fill_polygons(hdr, curves, GRID_RESOLUTION)
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
    for i in 1:min(PLOT_SAMPLES, size(samples.samples,2))
        x, y = xy(samples.samples[:,i])
        inside = samples.inside[i]
        push!(elements, dot(x, y; color=inside ? INK : ORANGE,
                            radius=inside ? 2.3 : 3.8, hollow=!inside))
    end
    for curve in curves
        push!(elements, path(xy.(curve); color="#0d1e15", width=2.4))
    end
    # Admitted simplex-edge portions are also part of the HDR boundary.
    for (a,b) in (([1.0,0,0],[0.0,1,0]), ([0.0,1,0],[0.0,0,1]), ([0.0,0,1],[1.0,0,0]))
        for i in 0:GRID_RESOLUTION-1
            t1, t2 = i / GRID_RESOLUTION, (i+1) / GRID_RESOLUTION
            q1, q2 = (1-t1).*a .+ t1.*b, (1-t2).*a .+ t2.*b
            if contains(hdr, (q1.+q2)./2)
                push!(elements, path([xy(q1), xy(q2)]; color="#0d1e15", width=2.0))
            end
        end
    end
    push!(elements, label(left, baseline + 50, PHI_LABELS[1]; size=12, align=hcenter))
    push!(elements, label(left + side, baseline + 50, PHI_LABELS[2]; size=12, align=hcenter))
    push!(elements, label(center, vertices[3][2] - 22, PHI_LABELS[3]; size=12, align=hcenter))
    push!(elements, label(center, 761, "tau = $(round(hdr.tau; sigdigits=5))";
                         size=10, align=hcenter))
    push!(elements, label(center, 787,
        "HDR coverage: $(round(100samples.coverage; digits=2))%"; size=10, align=hcenter))
    push!(elements, label(center, 813,
        "Corner labels: input counts; positions: probabilities"; size=10, align=hcenter))
    if all(==(1), hdr.distribution.alpha)
        push!(elements, label(center, 839, "Constant density: the full triangle is included";
                             size=9, align=hcenter))
    end
end

function make_figure(e, samples, curves)
    height = figure_height(e)
    elements = Any[poly([(0,0),(FIGURE_WIDTH,0),(FIGURE_WIDTH,height),(0,height)];color="white")]
    push!(elements,label(45,44,"Three-component Dirichlet: direct 95% HDR";size=22))
    push!(elements,label(45,82,
        "Direct PDF and threshold | epsilon = $(EPSILON) | K = $(K) | seed = $(SEED)";size=12))
    add_environment!(elements,e)
    add_simplex!(elements,e.hdr,samples,curves,560.0,"Dirichlet($(join(e.input_counts, ", ")))")
    push!(elements,poly([(45,881),(65,881),(65,897),(45,897)];color=GREEN,edge=INK))
    push!(elements,label(76,890,"95% HDR uncertainty region";size=10))
    push!(elements,path([(440,890),(471,890)];width=2.4))
    push!(elements,label(482,890,"f(θ; φ) = tau";size=10))
    push!(elements,dot(53,923))
    push!(elements,label(68,923,"Inside HDR";size=10))
    push!(elements,dot(447,923;color=ORANGE,radius=3.8,hollow=true))
    push!(elements,label(462,923,"Outside HDR";size=10))
    push!(elements,label(45,964,
        "No grouping: green dots have density at least tau; orange rings have density below tau.";size=11))
    push!(elements,label(45,1000,"Three illustrative outcomes for one given-step transition row:";size=11))
    for (i,(config,mode)) in enumerate(e.successors)
        text = "$(PHI_LABELS[i]) = $(e.input_counts[i]): r'=($(config.quad[1]),$(config.quad[2]),$(config.heading)), m'=$(join(Int.(mode)))"
        push!(elements,label(45,1030+28(i-1),text;size=11))
    end
    # Inputs containing ones can shade thousands of clipped grid triangles.
    # Compose's variadic compose() recurses once per child; attach them in a
    # loop to keep stack depth bounded and preserve the existing paint order.
    figure = context(units=UnitBox(0,0,FIGURE_WIDTH,height))
    for element in Iterators.reverse(elements)
        compose!(figure,element)
    end
    return figure
end

function write_report(e, samples)
    open(joinpath(OUTPUT_DIR,"report.txt"),"w") do io
        println(io,"Direct three-component Dirichlet 95% HDR; no grouping or projection")
        println(io,"Settings: K=$K, epsilon=$EPSILON, seed=$SEED, resolution=$GRID_RESOLUTION")
        println(io,"Source: r=$(e.state.config), m=$(e.state.mode), action=$ACTION")
        println(io,"Input counts: $(e.input_counts)")
        println(io,"Three illustrative outcomes, not the complete neighborhood support.")
        println(io,"No transition, count update, or episode is simulated.")
        for (i,(config,mode)) in enumerate(e.successors)
            println(io,"$(PHI_LABELS[i]) = $(e.input_counts[i]): r'=$config, m'=$mode")
        end
        println(io,"tau=$(e.hdr.tau)")
        println(io,"Held-out HDR coverage=$(samples.coverage)")
        println(io,"Region: f(θ;phi)>=tau. Boundary: f(θ;phi)=tau. Same test determines dot colors.")
    end
    open(joinpath(OUTPUT_DIR,"samples.csv"),"w") do io
        println(io,"theta_1,theta_2,theta_3,inside_hdr,density")
        for i in 1:min(PLOT_SAMPLES,VALIDATION_SAMPLES)
            p = samples.samples[:,i]
            println(io,join(vcat(p,[samples.inside[i],pdf(e.hdr.distribution,p)]),","))
        end
    end
end

function main()
    experiment = make_experiment()
    samples = validation_samples(experiment.hdr,SEED+1)
    curves = hdr_contours(experiment.hdr,GRID_RESOLUTION)
    run_checks(experiment,samples,curves)
    mkpath(OUTPUT_DIR)
    figure = make_figure(experiment,samples,curves)
    height = figure_height(experiment)
    draw(SVG(joinpath(OUTPUT_DIR,"hdr_simplex.svg"),FIGURE_WIDTH*px,height*px),figure)
    draw(PNG(joinpath(OUTPUT_DIR,"hdr_simplex.png"),FIGURE_WIDTH*px,height*px;dpi=120),figure)
    write_report(experiment,samples)
    println("Dirichlet input: ",experiment.input_counts)
    println("Held-out HDR coverage: ",samples.coverage)
    println("Saved PNG, SVG, samples.csv, and report.txt to ",OUTPUT_DIR)
    return experiment
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HDRThreeInputVisualizationTest.main()
end
