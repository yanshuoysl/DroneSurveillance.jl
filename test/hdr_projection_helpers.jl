# Test-only geometry. The full Dirichlet density and MC threshold are unchanged.

function validate_groups(groups, dimension)
    length(groups) == 3 || throw(ArgumentError("a ternary plot needs three groups"))
    all(group -> !isempty(group), groups) || throw(ArgumentError("groups cannot be empty"))
    sort(vcat(groups...)) == collect(1:dimension) ||
        throw(ArgumentError("groups must partition all successor indices exactly once"))
    return groups
end

"""Greedy balancing of count totals, with deterministic index-based tie breaking."""
function balanced_groups(counts)
    length(counts) >= 3 || throw(ArgumentError("at least three successors are needed"))
    groups = [Int[] for _ in 1:3]
    totals = zeros(3)
    for index in sortperm(collect(eachindex(counts)); by=i -> (-counts[i], i))
        group = argmin(totals)
        push!(groups[group], index)
        totals[group] += counts[index]
    end
    foreach(sort!, groups)
    return validate_groups(groups, length(counts))
end

group_probabilities(theta, groups) = [sum(theta[i] for i in group) for group in groups]

"""
Maximizer of the ORIGINAL density on the aggregation fiber for q.
For counts >= 1, each group allocates q[g] in proportion to counts[i]-1.
If all exponents in a group are zero, every allocation has the same density.
This uses no grouped PDF and estimates no second threshold.
"""
function maximizing_allocation(counts, groups, q)
    validate_groups(groups, length(counts))
    all(a -> isfinite(a) && a >= 1, counts) ||
        throw(ArgumentError("the projection formula requires finite counts >= 1"))
    length(q) == 3 && all(x -> isfinite(x) && 0 <= x <= 1, q) &&
        isapprox(sum(q), 1.0) || throw(ArgumentError("q must belong to the ternary simplex"))
    theta = zeros(length(counts))
    for (g, indices) in enumerate(groups)
        total = sum(counts[i] - 1 for i in indices)
        for i in indices
            theta[i] = total == 0 ? q[g] / length(indices) : q[g] * (counts[i] - 1) / total
        end
    end
    return theta
end

function profile_pdf(hdr, groups, q)
    theta = maximizing_allocation(hdr.distribution.alpha, groups, q)
    return pdf(hdr.distribution, theta)
end

projected_contains(hdr, groups, q) = profile_pdf(hdr, groups, q) >= hdr.tau

# q1 at the left vertex, q2 at the right, q3 at the top.
function ternary_xy(q, left, baseline, side)
    return (left + side * (q[2] + q[3] / 2), baseline - side * sqrt(3) / 2 * q[3])
end

"""Regular grid in (q1,q2), clipped to q1+q2 <= 1."""
function projection_grid(hdr, groups, resolution)
    resolution >= 10 || throw(ArgumentError("plot resolution must be at least 10"))
    axis = collect(range(0.0, 1.0; length=resolution + 1))
    field = zeros(length(axis), length(axis))
    for i in 0:resolution, j in 0:resolution-i
        q = [i / resolution, j / resolution, (resolution - i - j) / resolution]
        field[i + 1, j + 1] = profile_pdf(hdr, groups, q)
    end
    return axis, field
end

"""
Trace the convex projected region from its interior mode towards the simplex edges.
For positive group exponents, every ray has one boundary crossing. Bisection
uses the original PDF and tau directly, resolving contours close to the edges
without interpolating huge density values on a coarse grid.
"""
function radial_projection_boundary(hdr, groups, resolution)
    resolution >= 10 || throw(ArgumentError("plot resolution must be at least 10"))
    weights = [sum(hdr.distribution.alpha[i] - 1 for i in group) for group in groups]
    all(>(0), weights) || throw(ArgumentError("radial tracing requires positive group exponents"))
    center = weights ./ sum(weights)
    projected_contains(hdr, groups, center) || error("the projected mode must be inside the HDR")
    vertices = ([1.0,0,0], [0.0,1,0], [0.0,0,1])
    boundary = NTuple{3, Float64}[]
    for edge in 1:3, j in 0:resolution-1
        t = j / resolution
        target = (1-t) .* vertices[edge] .+ t .* vertices[mod1(edge+1, 3)]
        low, high = 0.0, 1.0
        for _ in 1:80
            middle = (low + high) / 2
            q = (1-middle) .* center .+ middle .* target
            if projected_contains(hdr, groups, q)
                low = middle
            else
                high = middle
            end
        end
        q = (1-low) .* center .+ low .* target
        push!(boundary, Tuple(q))
    end
    push!(boundary, first(boundary))
    return boundary
end

"""Trace the scalar level M(q)=tau; simplex-edge pieces are drawn separately."""
function projected_contours(hdr, groups, resolution)
    all(==(1), hdr.distribution.alpha) && return Vector{NTuple{3, Float64}}[]
    if all(sum(hdr.distribution.alpha[i] - 1 for i in group) > 0 for group in groups)
        return [radial_projection_boundary(hdr, groups, resolution)]
    end
    axis, field = projection_grid(hdr, groups, resolution)
    curves = Vector{NTuple{3, Float64}}[]
    for curve in Contour.lines(Contour.contour(axis, axis, field, hdr.tau))
        # Outside-simplex grid values are only a contouring mask. Discard those
        # coordinates rather than interpreting them as probability vectors.
        run = NTuple{3, Float64}[]
        for (x, y) in Contour.vertices(curve)
            if x >= 0 && y >= 0 && x + y <= 1 + 1e-12
                z = max(0.0, 1 - x - y)
                q = [x, y, z]
                q ./= sum(q)
                push!(run, Tuple(q))
            else
                length(run) > 1 && push!(curves, run)
                run = NTuple{3, Float64}[]
            end
        end
        length(run) > 1 && push!(curves, run)
    end
    return curves
end

"""Clip a small triangle to a linearly interpolated M(q)>=tau field."""
function clip_hdr_triangle(vertices, values, tau)
    polygon = NTuple{3, Float64}[]
    for i in 1:3
        j = mod1(i + 1, 3)
        inside_i, inside_j = values[i] >= tau, values[j] >= tau
        inside_i && push!(polygon, vertices[i])
        if inside_i != inside_j
            t = (tau - values[i]) / (values[j] - values[i])
            push!(polygon, ntuple(k -> (1 - t) * vertices[i][k] + t * vertices[j][k], 3))
        end
    end
    return polygon
end

function projected_fill_polygons(hdr, groups, resolution)
    all(==(1), hdr.distribution.alpha) &&
        return [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]]
    if all(sum(hdr.distribution.alpha[i] - 1 for i in group) > 0 for group in groups)
        return [radial_projection_boundary(hdr, groups, resolution)]
    end
    _, field = projection_grid(hdr, groups, resolution)
    polygons = Vector{NTuple{3, Float64}}[]
    q(i, j) = (i / resolution, j / resolution, (resolution - i - j) / resolution)
    function add_triangle(indices)
        vertices = [q(i, j) for (i, j) in indices]
        values = [field[i + 1, j + 1] for (i, j) in indices]
        clipped = clip_hdr_triangle(vertices, values, hdr.tau)
        length(clipped) >= 3 && push!(polygons, clipped)
    end
    for i in 0:resolution-1, j in 0:resolution-i-1
        add_triangle(((i, j), (i + 1, j), (i, j + 1)))
        if i + j <= resolution - 2
            add_triangle(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
        end
    end
    return polygons
end
