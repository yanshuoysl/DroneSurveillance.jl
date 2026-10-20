# Default Monte Carlo sample count. Edit K here, or pass n_samples per call.
const K = 10_000

# Density-quantile HDR: sample θ ~ Dir(φ), evaluate Y = f(θ; φ), and estimate
# τ_ε with Statistics.quantile(Y, ε), using its default linear interpolation.
# Reference: Hyndman (1996), Section 3.2, https://robjhyndman.com/papers/hdrgraph.pdf

function hdr_distribution(phi_row::AbstractVector{<:Real})
    isempty(phi_row) && throw(ArgumentError("a Dirichlet row cannot be empty"))
    counts = Float64.(phi_row)
    all(x -> isfinite(x) && x > 0, counts) ||
        throw(ArgumentError("Dirichlet counts must be finite and positive"))
    isfinite(sum(counts)) || throw(ArgumentError("sum of Dirichlet counts must be finite"))
    return Dirichlet(counts)
end

function check_hdr_settings(epsilon::Real, n_samples::Integer)
    isfinite(epsilon) && 0 < epsilon < 1 ||
        throw(ArgumentError("epsilon must be strictly between 0 and 1"))
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    return nothing
end

function estimate_tau(distribution::Dirichlet; epsilon::Real,
                      n_samples::Integer=K, rng::AbstractRNG=Random.default_rng())
    check_hdr_settings(epsilon, n_samples)
    # A single successor has probability 1: its simplex is the singleton {(1,)}.
    length(distribution) == 1 && return 1.0

    densities = Vector{Float64}(undef, n_samples)
    for i in eachindex(densities)
        theta = rand(rng, distribution)
        density = pdf(distribution, theta)
        isfinite(density) && density > 0 ||
            throw(DomainError(density, "sampled PDF must be finite and positive; cannot estimate tau reliably"))
        densities[i] = density
    end
    return quantile(densities, Float64(epsilon))
end

"""
    estimate_tau(phi_row; epsilon, n_samples=K, rng=Random.default_rng())

Estimate τ_ε for θ ~ Dir(phi_row) by Monte Carlo sampling of PDF values and
`quantile(Y, epsilon)` with Julia's default linear interpolation. `K` is the
editable default at the top of this file. A one-destination row is deterministic
and returns 1 without sampling.

The estimated threshold gives approximate posterior coverage, not a certified
finite-sample probability bound. Nonfinite or zero sampled PDFs raise an error
instead of silently returning a threshold affected by overflow or underflow.
"""
function estimate_tau(phi_row::AbstractVector{<:Real}; epsilon::Real,
                      n_samples::Integer=K, rng::AbstractRNG=Random.default_rng())
    return estimate_tau(hdr_distribution(phi_row); epsilon, n_samples, rng)
end

"""
    DirichletHDR

A continuous row uncertainty set Û_{ε,r,m,a}(φ) = {θ ∈ Δ : f(θ; φ_row) ≥ tau}.
Construct with `build_row_hdr`. `distribution.alpha` contains a snapshot of the
row counts, and `successors[i]` identifies the (r′, m′) pair for θ[i]. Treat these
stored arrays as read-only. Samples are used to estimate `tau`, not as a finite
representation of the uncertainty set.
"""
struct DirichletHDR{D<:Dirichlet}
    distribution::D
    successors::Vector{Tuple{DSState, FaultMode}}
    epsilon::Float64
    n_samples::Int
    tau::Float64
end

"""
    build_row_hdr(pomdp, s, action; epsilon, n_samples=K, rng=Random.default_rng())

Build an HDR over the complete action-dependent support S(r,m,a), using the
map-aware count-row API. Impossible destinations have no Dirichlet coordinate.
Source outcome labels must agree with the map; terminal sources have no row.
"""
function build_row_hdr(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                       action::Integer; epsilon::Real,
                       n_samples::Integer=K, rng::AbstractRNG=Random.default_rng())
    check_hdr_settings(epsilon, n_samples)
    source = support_source(pomdp, s.config, s.mode, action)
    s.outcome == source.outcome ||
        throw(ArgumentError("source outcome is inconsistent with its configuration"))
    destinations = feasible_successors(pomdp, s.config, s.mode, action)
    phi_row = count_row(pomdp, s.counts, s.config, s.mode, action, destinations)
    distribution = hdr_distribution(phi_row)
    tau = estimate_tau(distribution; epsilon, n_samples, rng)
    return DirichletHDR(distribution, destinations, Float64(epsilon), Int(n_samples), tau)
end

"""
    build_row_hdr(pomdp, s, action, successors;
                  epsilon, n_samples=K, rng=Random.default_rng())

Extract φ_{r,m,a} from hyperstate `s`, then estimate its HDR threshold. The caller
supplies the complete ordered support S(r,m,a), as `(next_config, next_mode)`
pairs. This low-level overload supports existing illustrative HDR fixtures: it
checks map validity, mode degradation, and duplicate keys, but does NOT validate
action-specific locomotion or completeness. Omit `successors` to construct a
model row with validated action-dependent support.
Unsafe destinations must use UNSAFE_CONFIG, not raw obstacle/outside positions.

Counts are read from this particular hyperstate and never updated here. Terminal
source states are absorbing and do not require a learned transition row.
"""
function build_row_hdr(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                       action::Integer, successors; epsilon::Real,
                       n_samples::Integer=K, rng::AbstractRNG=Random.default_rng())
    check_hdr_settings(epsilon, n_samples)
    canonical_source = adaptive_state(pomdp, s.config, s.mode, s.counts)
    is_terminal(canonical_source) &&
        throw(ArgumentError("HDR construction requires a nonterminal source state"))
    s.outcome == canonical_source.outcome ||
        throw(ArgumentError("source outcome is inconsistent with its configuration"))

    destinations = Tuple{DSState, FaultMode}[(config, mode) for (config, mode) in successors]
    for (config, mode) in destinations
        canonical = adaptive_state(pomdp, config, mode)
        config == canonical.config ||
            throw(ArgumentError("successors must be free grid configurations or UNSAFE_CONFIG"))
    end
    phi_row = count_row(s.counts, s.config, s.mode, action, destinations)
    distribution = hdr_distribution(phi_row)
    tau = estimate_tau(distribution; epsilon, n_samples, rng)
    return DirichletHDR(distribution, destinations, Float64(epsilon), Int(n_samples), tau)
end

"""
    contains(hdr::DirichletHDR, theta)

Test simplex membership and f(theta; φ_row) ≥ tau. Probability normalization
uses Julia's default `isapprox` tolerance. Also available as `theta in hdr`.
"""
function Base.contains(hdr::DirichletHDR, theta::AbstractVector{<:Real})
    length(theta) == length(hdr.successors) || return false
    all(x -> isfinite(x) && 0 <= x <= 1, theta) || return false
    isapprox(sum(theta), 1.0) || return false
    return pdf(hdr.distribution, theta) >= hdr.tau
end

Base.in(theta::AbstractVector{<:Real}, hdr::DirichletHDR) = contains(hdr, theta)
