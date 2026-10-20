# Run from the repository root:
# julia --project=test test/test_hdr_visualization.jl
# Standalone supplied-parameter HDR test; no environment or count simulation.
module HDRVisualizationTest

using Random

# ---- Editable experiment settings -------------------------------------------
const K = 10_000
const EPSILON = 0.05
const SEED = 42
# Edit this list to specify phi directly. Its length determines the dimension.
# Real values >= 1 are supported by the existing projection method.
const DIRICHLET_INPUT = [4.0, 10.0, 7.0, 9.0, 7.0]
const GRID_RESOLUTION = 160
const VALIDATION_SAMPLES = 10_000
const PLOT_SAMPLES = 450
# :density separates counts > 1 from counts == 1, instead of equalizing totals.
# :balanced restores the previous grouping. Neither guarantees class separation.
const GROUPING = :density
# Set three explicit index lists to override GROUPING, or leave nothing.
# Exactly three parameters use [[1],[2],[3]] directly unless overridden here.
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
    length(counts) >= 3 || throw(ArgumentError("at least three parameters are needed"))
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

"""The supplied Dirichlet distribution and its continuous HDR threshold.
This test-only container has no robot state, hyperstate, or successor metadata.
"""
struct ParameterHDR{D}
    distribution::D
    epsilon::Float64
    n_samples::Int
    tau::Float64
end

function Base.contains(hdr::ParameterHDR, theta::AbstractVector{<:Real})
    length(theta) == length(hdr.distribution) || return false
    all(x -> isfinite(x) && 0 <= x <= 1, theta) || return false
    isapprox(sum(theta),1.0) || return false
    return pdf(hdr.distribution,theta) >= hdr.tau
end

Base.in(theta::AbstractVector{<:Real}, hdr::ParameterHDR) = contains(hdr,theta)

function validate_parameters(input)
    input isa AbstractVector || throw(ArgumentError("DIRICHLET_INPUT must be a vector"))
    length(input) >= 3 || throw(ArgumentError("at least three Dirichlet parameters are needed"))
    all(x -> x isa Real && isfinite(x) && x >= 1, input) ||
        throw(ArgumentError("DIRICHLET_INPUT must contain finite real values >= 1; the current projection does not support values below 1"))
    return Float64.(input)
end

function select_groups(parameters, groups_input, grouping)
    if !isnothing(groups_input)
        return validate_groups(deepcopy(groups_input),length(parameters)), :manual
    end
    grouping in (:density,:balanced) ||
        throw(ArgumentError("GROUPING must be :density or :balanced"))
    # No information needs to be discarded when the input already has size 3.
    length(parameters) == 3 && return [[1],[2],[3]], :direct
    return grouping == :density ? (density_groups(parameters), :density) :
                                  (balanced_groups(parameters), :balanced)
end

function make_experiment(; dirichlet_input=DIRICHLET_INPUT, groups_input=GROUPS,
                         grouping=GROUPING, n_samples=K, epsilon=EPSILON, seed=SEED)
    parameters = validate_parameters(dirichlet_input)
    groups, selected_grouping = select_groups(parameters,groups_input,grouping)
    # Reuse the original PDF-based Monte Carlo estimator, without creating a
    # drone environment or putting these parameters into a simulated hyperstate.
    distribution = DS.hdr_distribution(parameters)
    tau = estimate_tau(distribution; epsilon, n_samples, rng=MersenneTwister(seed))
    hdr = ParameterHDR(distribution,Float64(epsilon),Int(n_samples),tau)
    return (; parameters, groups, hdr, grouping=selected_grouping, seed)
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
    candidates = length(e.parameters) == 3 ? [(:direct,[[1],[2],[3]])] :
        [(:balanced,balanced_groups(e.parameters)),(:density,density_groups(e.parameters))]
    e.grouping == :manual && push!(candidates,(:manual,e.groups))
    return [let
        included = [projected_contains(e.hdr, groups, group_probabilities(theta, groups))
                    for theta in eachcol(samples.samples)]
        outside = .!samples.inside_full
        (; strategy, totals=[sum(e.parameters[g]) for g in groups],
           orange_total=sum(outside), orange_outside=sum(outside .& .!included),
           projected_coverage=sum(included)/length(included))
    end for (strategy, groups) in candidates]
end

function run_checks(e, samples, curves)
    @testset "Supplied-parameter Dirichlet HDR" begin
        @test e.hdr.distribution.alpha == e.parameters
        @test length(e.parameters) >= 3
        @test size(samples.samples) == (length(e.parameters),VALIDATION_SAMPLES)
        @test sort(vcat(e.groups...)) == collect(eachindex(e.parameters))
        @test all(group -> !isempty(group),e.groups)
        @test all(isfinite,samples.densities)
        @test all(s -> 0 <= s <= 1,samples.scores)
        @test (samples.scores .>= 0.5) == samples.inside_full
        @test e.hdr.tau == estimate_tau(e.parameters; epsilon=e.hdr.epsilon,
            n_samples=e.hdr.n_samples,rng=MersenneTwister(e.seed))
        @test samples.samples == rand(MersenneTwister(e.seed+1),e.hdr.distribution,VALIDATION_SAMPLES)
        if all(==(1),e.parameters)
            @test samples.coverage == 1.0
            @test samples.projected_coverage == 1.0
        else
            @test abs(samples.coverage-(1-e.hdr.epsilon)) < 0.025
        end
        for i in 1:VALIDATION_SAMPLES
            @test isapprox(sum(samples.grouped[i]),1)
            @test !samples.inside_full[i] || samples.inside_projection[i]
        end
        for i in 1:min(100,VALIDATION_SAMPLES)
            value = profile_pdf(e.hdr,e.groups,samples.grouped[i])
            original = samples.densities[i]
            @test value >= original || isapprox(value,original;rtol=1e-12)
        end
        if length(e.parameters) == 3
            # Singleton groups preserve full membership, including permutations.
            @test samples.inside_projection == samples.inside_full
        end
        if all(sum(e.parameters[i]-1 for i in group) > 0 for group in e.groups)
            @test !isempty(curves)
            for curve in curves, point in curve
                @test isapprox(profile_pdf(e.hdr,e.groups,collect(point)),e.hdr.tau;rtol=0.04)
            end
        end

        # Fractional parameters remain exact; external input edits cannot alter
        # the constructed distribution or its stored parameter snapshot.
        input = [1.5,2.25,3.0,1.0]
        fractional = make_experiment(dirichlet_input=input,groups_input=nothing,n_samples=100)
        input[1] = 9.0
        @test fractional.parameters == [1.5,2.25,3.0,1.0]
        @test fractional.hdr.distribution.alpha == fractional.parameters
        @test fractional.hdr.tau == estimate_tau(fractional.parameters;epsilon=EPSILON,
            n_samples=100,rng=MersenneTwister(SEED))
        direct = make_experiment(dirichlet_input=[2,3,2],groups_input=nothing,n_samples=100)
        @test direct.groups == [[1],[2],[3]]
        @test direct.grouping == :direct
        @test contains(direct.hdr,[0.25,0.5,0.25])
        @test !contains(direct.hdr,[0.5,0.5])
        @test !contains(direct.hdr,[0.5,0.5,0.5])
        @test !contains(direct.hdr,[-0.1,0.5,0.6])
        @test !contains(direct.hdr,[NaN,0.5,0.5])
        @test [0.25,0.5,0.25] in direct.hdr
        reordered = make_experiment(dirichlet_input=[2,3,2],groups_input=[[2],[1],[3]],n_samples=100)
        @test reordered.groups == [[2],[1],[3]]
        @test reordered.grouping == :manual
        for invalid in (nothing,[2,3],[2,0,2],[2,-1,2],[2,0.5,2],
                        [2,Inf,2],[2,NaN,2],[2,"three",2])
            @test_throws ArgumentError make_experiment(dirichlet_input=invalid,groups_input=nothing,n_samples=100)
        end
        @test_throws ArgumentError make_experiment(dirichlet_input=[2,3,2],
            groups_input=[[1],[2],[2,3]],n_samples=100)
        @test_throws ArgumentError make_experiment(dirichlet_input=[2,3,2],
            groups_input=nothing,grouping=:unknown,n_samples=100)
        @test density_groups([2,2,1]) == [[1],[2],[3]]
        @test density_groups([2,1,1,1]) == [[1],[2,4],[3]]
        @test density_groups(ones(6)) == balanced_groups(ones(6))
        @test density_score(0.0,2.0) == 0.0
        @test density_score(2.0,2.0) == 0.5
        @test density_score(1e300,1e300) == 0.5

        # Independently check the projection maximizer on a supplied 5D case.
        alpha = [3.0,5.0,2.0,4.0,6.0]
        groups = [[1],[2],[3,4,5]]
        q = [0.25,0.25,0.5]
        distribution = DS.Dirichlet(alpha)
        allocation = maximizing_allocation(alpha,groups,q)
        analytic = pdf(distribution,allocation)
        brute_force = maximum(pdf(distribution,
            [q[1],q[2],0.5i/100,0.5j/100,0.5(100-i-j)/100])
            for i in 0:100 for j in 0:100-i)
        @test group_probabilities(allocation,groups) ≈ q
        @test analytic >= brute_force || isapprox(analytic,brute_force)
        @test isapprox(analytic,brute_force;rtol=0.005)
        @test maximizing_allocation(ones(5),groups,q)[3:5] ≈ fill(1/6,3)
        @test_throws ArgumentError validate_groups([[1],[2],[2,3]],3)
        @test_throws ArgumentError validate_groups([[1],[2],Int[]],2)
        @test_throws ArgumentError maximizing_allocation(alpha,groups,[0.3,0.3,0.3])
        @test_throws ArgumentError maximizing_allocation([0.5,2,2],[[1],[2],[3]],q)
        uniform = make_experiment(dirichlet_input=ones(length(e.parameters)),
                                  groups_input=e.groups,n_samples=100)
        @test isempty(projected_contours(uniform.hdr,e.groups,20))
        @test all(projected_contains(uniform.hdr,e.groups,q)
                  for q in ([1.0,0,0],[0.0,1,0],[0.0,0,1],[1/3,1/3,1/3]))
    end
end

# Compose draws the figure; the geometry helper traces M(q)=tau.
const FIGURE_WIDTH = 1200
const FIGURE_HEIGHT = 1150
figure_height(e) = max(FIGURE_HEIGHT, 1090 + 23cld(min(length(e.parameters), 10), 2))
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

function add_parameter_summary!(elements, e)
    push!(elements,label(45,156,"Supplied Dirichlet parameters";size=14))
    for i in 1:min(12,length(e.parameters))
        push!(elements,label(45,200+25(i-1),"phi[$i] = $(e.parameters[i])";size=12))
    end
    if length(e.parameters) > 12
        push!(elements,label(45,506,"More entries in report.txt";size=10))
    end
    push!(elements,label(45,535,"Dimension: $(length(e.parameters))";size=12))
    push!(elements,label(45,563,"Parameter sum: $(sum(e.parameters))";size=12))
    push!(elements,label(45,605,"USER-SUPPLIED INPUT";size=11,color="#92501e"))
    for (i,text) in enumerate(("Parameters read directly from the list.",
                               "No environment or generated counts.",
                               "Grouping is for visualization only."))
        push!(elements,label(45,634+23(i-1),text;size=10))
    end
end

function add_simplex!(elements, hdr, groups, samples, curves, left, title)
    baseline, side = 670.0, 485.0
    xy(q) = ternary_xy(q, left, baseline, side)
    vertices = xy.(((1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)))
    center = left + side / 2
    push!(elements, label(center, 156, title; size=15, align=hcenter))
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
    coordinates = ["<i>q</i><sub>$(i)</sub>" for i in 1:3]
    push!(elements, label(left, baseline + 50, coordinates[1]; size=12, align=hcenter))
    push!(elements, label(left + side, baseline + 50, coordinates[2]; size=12, align=hcenter))
    push!(elements, label(center, vertices[3][2] - 22, coordinates[3]; size=12, align=hcenter))
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
    push!(elements, label(45, 44, length(e.parameters) == 3 ? "Given Dirichlet parameters: direct 95% HDR" : "Given Dirichlet parameters: projected 95% HDR"; size=22))
    push!(elements, label(45, 82,
        "Original $(length(e.parameters))-component PDF and threshold | epsilon = $(e.hdr.epsilon) | K = $(e.hdr.n_samples) | seed = $(e.seed)"; size=12))
    add_parameter_summary!(elements, e)
    add_simplex!(elements, e.hdr, e.groups, samples, curves, 560.0,
        "Dirichlet($(join(e.parameters, ", ")))")
    push!(elements, label(45, 738,
        "Parameters: $(count(==(1), e.parameters)) equal to one"; size=11))
    push!(elements, label(45, 764,
        "Group totals: " * join([sum(e.parameters[g]) for g in e.groups], ", "); size=11))
    if e.grouping == :density && all(e.parameters[i] > 1 for g in e.groups[1:2] for i in g) &&
       all(e.parameters[i] == 1 for i in e.groups[3])
        push!(elements, label(45, 791, "q1, q2: split parameters above one"; size=10))
        push!(elements, label(45, 815, "q3: parameter-one entries"; size=10))
    end

    push!(elements, poly([(45,881),(65,881),(65,897),(45,897)]; color=GREEN, edge=INK))
    push!(elements, label(76, 890, length(e.parameters) == 3 ? "HDR uncertainty region" : "Projected uncertainty region"; size=10))
    push!(elements, path([(440,890),(471,890)]; width=2.4))
    push!(elements, label(482, 890, length(e.parameters) == 3 ? "f(theta; phi) = tau" : "M(q) = original tau"; size=10))
    push!(elements, dot(53,923))
    push!(elements, label(68, 923, "Inside original HDR"; size=10))
    push!(elements, dot(447,923; color=ORANGE, radius=3.8, hollow=true))
    push!(elements, label(462, 923, "Outside original HDR"; size=10))
    push!(elements, label(45, 964,
        length(e.parameters) == 3 ? "No aggregation: the region and point colors use the same PDF threshold." :
        "Grouping hides allocations: an orange point may lie inside green."; size=11))
    group_text = join(["q$(g) = " * join(indices, "+") for (g,indices) in enumerate(e.groups)], "     |     ")
    length(group_text) > 100 && (group_text = "See report.txt for all group memberships")
    push!(elements, label(45, 995, "Groups (parameter indices): " * group_text; size=11))
    push!(elements,label(45,1025,
        "Parameter indices and values (all entries in report.txt):";size=10))
    for i in 1:min(10,length(e.parameters))
        row,column = divrem(i-1,2)
        push!(elements,label(45+590column,1055+23row,
            "phi[$i] = $(e.parameters[i])";size=10))
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
        "Same $(count) samples as simplex | original $(length(e.parameters))-component PDF | epsilon = $(EPSILON)"; size=12))
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
        println(io, "Supplied-parameter Dirichlet 95% HDR; no environment or generated parameters")
        println(io, "Settings: K=$(e.hdr.n_samples), epsilon=$(e.hdr.epsilon), seed=$(e.seed), resolution=$GRID_RESOLUTION")
        println(io, "Dimension: $(length(e.parameters))")
        println(io, "Input parameters: $(e.parameters)")
        println(io, length(e.parameters) == 3 ? "Direct three-component visualization; no aggregation." :
            "Original full-dimensional HDR, projected onto three groups for visualization.")
        println(io, "Selected grouping strategy: $(e.grouping)")
        println(io, "Group parameter totals: $([sum(e.parameters[g]) for g in e.groups])")
        println(io, "Groups: $(e.groups)")
        for (i, value) in enumerate(e.parameters)
            println(io, "Parameter $i: phi[$i]=$value")
        end
        println(io, "Original tau=$(e.hdr.tau)")
        println(io, "Held-out original coverage=$(samples.coverage), projected coverage=$(samples.projected_coverage)")
        println(io, "Grouping comparison on the SAME independent samples (not used to choose the grouping):")
        for c in comparison
            println(io, "  $(c.strategy): totals=$(c.totals), orange outside projection=$(c.orange_outside)/$(c.orange_total), projected coverage=$(c.projected_coverage)")
        end
        if length(e.parameters) > 3
            println(io, "Complete green/orange separation is not guaranteed by a three-group projection.")
        end
        println(io, "hdr_density.png plots the original density score f/(f+tau), with the SAME membership cutoff at 0.5. This display score is not a probability.")
    end
    open(joinpath(OUTPUT_DIR, "samples.csv"), "w") do io
        println(io, join(vcat(["q1", "q2", "q3", "inside_full_hdr", "inside_projection", "density", "density_score"],
                             ["theta_$i" for i in eachindex(e.parameters)]), ","))
        for i in 1:min(PLOT_SAMPLES, VALIDATION_SAMPLES)
            println(io, join(vcat(samples.grouped[i],
                [samples.inside_full[i], samples.inside_projection[i], samples.densities[i], samples.scores[i]], samples.samples[:,i]), ","))
        end
    end
end

function main()
    experiment = make_experiment()
    samples = validation_samples(experiment.hdr, experiment.groups, experiment.seed + 1)
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
    println("Supplied Dirichlet parameters: ", experiment.parameters)
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
