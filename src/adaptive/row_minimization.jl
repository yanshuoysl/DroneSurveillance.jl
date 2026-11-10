# Linear minimization over one fixed Dirichlet HDR. No model/count update.

"""A row solve could not produce a verified minimum within its numerical limits."""
struct RowMinimizationError <: Exception
    message::String
end
Base.showerror(io::IO, e::RowMinimizationError) = print(io, e.message)

"""
Result of `minimize_hdr_row`. `theta` follows the supplied HDR's successor order.
`value` is the expected successor value, and `objective_lower_bound` and
`optimality_gap` give a numerical lower bound and gap from convexity/KKT checks.
These diagnostics use Float64 arithmetic, not interval-certified bounds.

`kappa` and `lambda` are in the original value units. `shifted_kappa` stores
kappa + minimum(successor_values) without subtractive cancellation. If Float64
would round kappa onto its forbidden pole, kappa is reported as a BigFloat;
root finding and theta still use Float64 and the positive shifted coordinate.
No kappa is reported on direct exits; lambda is zero for constant values and
`nothing` for singleton/degenerate sets, where no root is needed.
`status` is :converged, :constant_values, :singleton, or :mode_only. Every return
is successful; invalid inputs and failed convergence throw explicitly.
"""
struct HDRRowMinimum
    theta::Vector{Float64}
    value::Float64
    kappa::Union{Nothing,Float64,BigFloat}
    lambda::Union{Nothing,Float64}
    shifted_kappa::Union{Nothing,Float64}
    iterations::Int
    bracket_iterations::Int
    status::Symbol
    log_hdr_slack::Float64
    stationarity_residual::Union{Nothing,Float64}
    relative_stationarity_residual::Union{Nothing,Float64}
    optimality_gap::Float64
    objective_lower_bound::Float64
end

function row_reported_kappa(value_scale, shift, value_min)
    kappa = muladd(value_scale, shift, -value_min)
    isfinite(kappa) && kappa > -value_min && return kappa
    # Reporting kappa must preserve the strict domain even when the valid
    # positive shift is smaller than one Float64 ulp of -minimum(v).
    bits = value_min == 0 ? 256 : max(256,
        128 + exponent(abs(value_min)) - exponent(value_scale) - exponent(shift))
    return setprecision(BigFloat, bits) do
        BigFloat(value_scale) * BigFloat(shift) - BigFloat(value_min)
    end
end

function row_probabilities_at_shift(d, values, shift)
    # Original formula: theta_j = (d_j/(v_j+kappa))/sum_l(d_l/(v_l+kappa)).
    # Here values are shifted/scaled, and shift is strictly positive.
    denominators = values .+ shift
    logweights = log.(d) .- log.(denominators)
    weights = exp.(logweights .- maximum(logweights))
    theta = weights ./ sum(weights)
    loglambda = -(maximum(logweights) + log(sum(weights)))
    return (; theta, lambda=exp(loglambda), shift)
end

function row_checked_probabilities(hdr, theta)
    # Check the normalization used by the transition/selected-row interfaces too.
    all(>(0), theta) || return false
    contains(hdr, theta) || return false
    next = transition_probabilities(theta, length(theta))
    contains(hdr, next) || return false
    return contains(hdr, transition_probabilities(next, length(next)))
end

function row_direct_minimum(hdr, distribution, theta, values, status)
    row_checked_probabilities(hdr, theta) ||
        throw(RowMinimizationError("the direct solution cannot be represented inside the supplied HDR"))
    value = sum(values .* theta)
    isfinite(value) || throw(RowMinimizationError("row objective exceeds Float64 range"))
    slack = Distributions.logpdf(distribution, theta) - log(hdr.tau)
    return HDRRowMinimum(copy(theta), value, nothing,
                         status == :constant_values ? 0.0 : nothing, nothing,
                         0, 0, status, slack, nothing, nothing, 0.0, value)
end

function row_minimum_candidate(hdr, distribution, candidate, d, values, scaled_values,
                               value_scale, value_min, iterations, bracket_iterations,
                               log_atol, objective_atol, objective_rtol, kkt_rtol)
    theta = candidate.theta
    slack = Distributions.logpdf(distribution, theta) - log(hdr.tau)
    isfinite(slack) && 0 <= slack <= log_atol || return nothing
    row_checked_probabilities(hdr, theta) || return nothing
    lambda = candidate.lambda
    isfinite(lambda) && lambda > 0 || return nothing
    terms = (lambda ./ theta) .* d
    residuals = scaled_values .+ candidate.shift .- terms
    all(isfinite, residuals) || return nothing
    residual = maximum(abs, residuals)
    relative_residual = residual / max(1.0, maximum(scaled_values) + candidate.shift,
                                      maximum(abs, terms))
    relative_residual <= kkt_rtol || return nothing

    # For L(theta)=v'theta-lambda*G(theta), convexity gives a lower bound
    # L(theta)+min_j(gradient_j)-gradient'theta over the simplex. Thus this gap
    # includes both complementary slackness and the stationarity error.
    gradient_gap = max(0.0, sum(theta .* residuals) - minimum(residuals))
    scaled_gap = lambda * slack + gradient_gap
    scaled_tolerance = objective_atol / value_scale + objective_rtol * maximum(scaled_values)
    isfinite(scaled_gap) && scaled_gap <= scaled_tolerance || return nothing

    value = sum(values .* theta)
    gap = value_scale * scaled_gap
    kappa = row_reported_kappa(value_scale, candidate.shift, value_min)
    original_lambda = value_scale * lambda
    shifted_kappa = value_scale * candidate.shift
    stationarity = value_scale * residual
    all(isfinite, (value, gap, kappa, original_lambda, shifted_kappa, stationarity, value-gap)) ||
        throw(RowMinimizationError("minimum or its diagnostics exceed Float64 range"))
    original_lambda > 0 && shifted_kappa > 0 ||
        throw(RowMinimizationError("positive KKT multipliers underflow Float64"))
    return HDRRowMinimum(copy(theta), value, kappa, original_lambda, shifted_kappa,
                         iterations, bracket_iterations, :converged, slack,
                         stationarity, relative_residual, gap, value-gap)
end

"""
    minimize_hdr_row(hdr, successor_values;
        log_atol=1e-9, objective_atol=1e-10, objective_rtol=1e-9,
        kkt_rtol=1e-10, max_iterations=256, max_bracket_iterations=1024)

Minimize dot(successor_values, theta) over the *supplied*, fixed DirichletHDR.
Values must be finite and follow hdr.successors order. Learned-row parameters
must exceed one (the model starts at two). Generic HDR helpers still support
other positive parameters; mixed/zero-exponent optimization is not handled here.

For nonconstant values, implement the slides' closed form
theta_j(kappa) ∝ (phi_j-1)/(v_j+kappa), kappa > -minimum(v), and bisect
G(kappa)=logpdf(Dir(phi),theta(kappa))-log(tau)=0. This is exactly
sum_j((phi_j-1)*log(theta_j))-c, c=log(tau)+log(B(phi)). Work with shifted,
positively scaled values to avoid cancellation at the pole. Thresholds are never
resampled. The feasible endpoint is returned only after HDR, stationarity,
complementarity, and objective-gap checks pass.

Equal values return the mode; singleton and mode-only HDRs return their unique
point. An empty/malformed HDR raises ArgumentError; numerical or convergence
failure raises RowMinimizationError. There is no approximate fallback. This
routine chooses one row for fixed coefficients, not the full POMDP policy or
continuation values. It never changes counts, inputs, or RNG state. Parent/map
compatibility is subsequently checked by the existing transition/row interfaces.
"""
function minimize_hdr_row(hdr::DirichletHDR, successor_values::AbstractVector{<:Real};
                          log_atol::Real=1e-9, objective_atol::Real=1e-10,
                          objective_rtol::Real=1e-9, kkt_rtol::Real=1e-10,
                          max_iterations::Integer=256, max_bracket_iterations::Integer=1024)
    for (name, tolerance) in ((:log_atol, log_atol), (:objective_atol, objective_atol),
                              (:objective_rtol, objective_rtol), (:kkt_rtol, kkt_rtol))
        isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("$name must be finite and nonnegative"))
    end
    log_atol > 0 && kkt_rtol > 0 || throw(ArgumentError("log_atol and kkt_rtol must be positive"))
    objective_atol > 0 || objective_rtol > 0 ||
        throw(ArgumentError("at least one objective tolerance must be positive"))
    max_iterations > 0 && max_bracket_iterations > 0 ||
        throw(ArgumentError("iteration limits must be positive"))
    check_hdr_settings(hdr.epsilon, hdr.n_samples)
    isfinite(hdr.tau) && hdr.tau > 0 || throw(ArgumentError("HDR threshold must be finite and positive"))
    n = length(hdr.successors)
    n > 0 && length(hdr.distribution) == n && length(successor_values) == n ||
        throw(ArgumentError("counts, successors, and values must have the same nonzero length"))
    length(unique(hdr.successors)) == n || throw(ArgumentError("HDR successors must be distinct"))
    values = Float64.(successor_values)
    all(isfinite, values) || throw(ArgumentError("successor values must be finite Float64 numbers"))
    phi_row = Float64.(hdr.distribution.alpha)
    all(x -> isfinite(x) && x > (n == 1 ? 0 : 1), phi_row) ||
        throw(ArgumentError("row minimization requires phi_j > 1 (except a singleton)"))
    distribution = hdr_distribution(phi_row)
    if n == 1
        hdr.tau <= 1 || throw(ArgumentError("singleton HDR is empty: tau exceeds one"))
        return row_direct_minimum(hdr, distribution, [1.0], values, :singleton)
    end
    d = phi_row .- 1
    mode = d ./ sum(d)
    mode ./= sum(mode)
    all(>(0), mode) || throw(RowMinimizationError("Dirichlet mode underflows Float64"))
    log_mode_pdf = Distributions.logpdf(distribution, mode)
    isfinite(log_mode_pdf) || throw(ArgumentError("Dirichlet log density is not finite at its mode"))
    isapprox(Distributions.logpdf(hdr.distribution, mode), log_mode_pdf; atol=1e-12, rtol=32eps()) ||
        throw(ArgumentError("HDR distribution has inconsistent cached parameters"))
    mode_pdf = pdf(distribution, mode)
    hdr.tau <= mode_pdf || throw(ArgumentError("HDR is empty: threshold exceeds maximum density"))
    if hdr.tau == mode_pdf
        return row_direct_minimum(hdr, distribution, mode, values, :mode_only)
    elseif all(==(first(values)), values)
        return row_direct_minimum(hdr, distribution, mode, values, :constant_values)
    end

    value_min, value_max = extrema(values)
    value_scale = value_max - value_min
    scaled_values = if isfinite(value_scale)
        (values .- value_min) ./ value_scale
    else
        # Avoid overflowing a difference of two individually finite values.
        value_scale = max(abs(value_min), abs(value_max))
        values ./ value_scale .- value_min / value_scale
    end
    log_tau = log(hdr.tau)
    evaluate(shift) = begin
        candidate = row_probabilities_at_shift(d, scaled_values, shift)
        slack = Distributions.logpdf(distribution, candidate.theta) - log_tau
        isnan(slack) && throw(RowMinimizationError("nonfinite root residual"))
        (; candidate..., slack)
    end

    # Find a genuine sign bracket in the positive shifted domain, rather than
    # inserting a fixed epsilon above -minimum(v).
    initial = evaluate(1.0)
    bracket_iterations = 0
    if initial.slack >= 0
        upper = initial
        lower = initial
        while lower.slack >= 0
            bracket_iterations < max_bracket_iterations ||
                throw(RowMinimizationError("failed to bracket the HDR root near its lower domain boundary"))
            upper = lower
            next_shift = lower.shift / 2
            next_shift > 0 || throw(RowMinimizationError("HDR root is below the representable positive shift"))
            lower = evaluate(next_shift)
            bracket_iterations += 1
        end
    else
        lower = initial
        upper = initial
        while upper.slack < 0
            bracket_iterations < max_bracket_iterations ||
                throw(RowMinimizationError("failed to bracket the HDR root toward the Dirichlet mode"))
            lower = upper
            next_shift = upper.shift * 2
            isfinite(next_shift) || throw(RowMinimizationError("HDR root exceeds the representable positive shift"))
            upper = evaluate(next_shift)
            bracket_iterations += 1
        end
    end

    for iteration in 0:max_iterations
        result = row_minimum_candidate(hdr, distribution, upper, d, values, scaled_values,
            value_scale, value_min, iteration, bracket_iterations, log_atol,
            objective_atol, objective_rtol, kkt_rtol)
        result === nothing || return result
        iteration == max_iterations && break
        midpoint = lower.shift + (upper.shift - lower.shift) / 2
        lower.shift < midpoint < upper.shift ||
            throw(RowMinimizationError("bisection reached Float64 resolution before verification passed"))
        candidate = evaluate(midpoint)
        if candidate.slack >= 0
            upper = candidate
        else
            lower = candidate
        end
    end
    throw(RowMinimizationError("HDR row minimization did not converge in $max_iterations bisection iterations"))
end
