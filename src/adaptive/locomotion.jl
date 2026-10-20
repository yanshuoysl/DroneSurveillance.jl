# First transition factor: T_theta(r',m' | r,m,phi,a). No count updates.

function transition_source(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                           action::Integer)
    source = support_source(pomdp, s.config, s.mode, action)
    s.outcome == source.outcome ||
        throw(ArgumentError("source outcome is inconsistent with its configuration"))
    return source
end

function transition_probabilities(theta::AbstractVector{<:Real}, n::Integer)
    length(theta) == n ||
        throw(ArgumentError("transition row needs one probability per feasible successor ($n)"))
    probabilities = Float64.(theta)
    all(p -> isfinite(p) && 0 <= p <= 1, probabilities) ||
        throw(ArgumentError("transition probabilities must be finite and between 0 and 1"))
    total = sum(probabilities)
    isapprox(total, 1.0; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("transition probabilities must sum to 1"))
    # Copy and normalize accepted roundoff; never modify the caller's vector.
    return probabilities ./ total
end

"""
    locomotion_mode_uncertainty(pomdp, s::AdaptiveState, action;
                                epsilon, n_samples=K, rng=Random.default_rng())

Represent the AR-POMDP's uncertain locomotion/mode row as a DirichletHDR over
S(r,m,a), with alpha extracted from this hyperstate's own transition counts.
The result is the continuous set U(phi), not a posterior-mean probability row
or one sampled transition model. Its successors retain the full Cartesian
configuration/mode support. Monte Carlo samples only calibrate the existing
HDR threshold, giving approximate posterior coverage rather than a certificate.

This reads phi without changing it. Goal/unsafe sources have no uncertain row
and raise ArgumentError. No motion, row selection, or parameter update occurs.
"""
function locomotion_mode_uncertainty(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                                     action::Integer; epsilon::Real,
                                     n_samples::Integer=K,
                                     rng::AbstractRNG=Random.default_rng())
    transition_source(pomdp, s, action)
    return build_row_hdr(pomdp, s, action; epsilon, n_samples, rng)
end

"""
    locomotion_mode_transition(pomdp, s::AdaptiveState, action, theta; hdr=nothing)

Instantiate ONLY T_theta(r',m' | r,m,phi,a). For an active source, theta must
belong to the supplied DirichletHDR representing U(phi). The HDR must contain
every pair in S(r,m,a) exactly once and match the current row counts. Theta
uses hdr.successors order; complete reorderings are supported. Malformed,
incomplete, stale, or unrelated HDRs and inadmissible probability rows are rejected.

Return a SparseCat over `(DSState, FaultMode)` pairs, with probability theta[j]
on hdr.successors[j] and zero on other pairs. No AdaptiveState or phi' is
constructed, and no count is incremented. Creating or sampling this distribution
does not mutate s.counts, theta, or hdr. Arrays in the returned distribution are
independent snapshots. Fault flags do not select a deterministic motion outcome.

There is no default posterior-mean row: the caller supplies an admissible theta.
Worst-case row selection belongs in the planner, not in this transition factor.
The existing HDR threshold is reused, without new calibration samples here.

Goal/unsafe sources are absorbing: pass theta=[1] without an HDR to get
Deterministic((s.config,s.mode)). Such sources have no learned uncertainty set.
"""
function locomotion_mode_transition(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                                    action::Integer, theta::AbstractVector{<:Real};
                                    hdr::Union{Nothing,DirichletHDR}=nothing)
    source = transition_source(pomdp, s, action)
    if is_terminal(source)
        hdr === nothing || throw(ArgumentError("terminal sources have no uncertain transition row"))
        transition_probabilities(theta, 1)
        return Deterministic((s.config, s.mode))
    end
    hdr === nothing && throw(ArgumentError("active sources require a DirichletHDR representing U(phi)"))
    counts = count_row(pomdp, s.counts, s.config, s.mode, action, hdr.successors)
    hdr.distribution.alpha == Float64.(counts) ||
        throw(ArgumentError("HDR parameters do not match the current transition row counts"))
    check_hdr_settings(hdr.epsilon, hdr.n_samples)
    isfinite(hdr.tau) && hdr.tau > 0 ||
        throw(ArgumentError("HDR threshold must be finite and positive"))
    probabilities = transition_probabilities(theta, length(hdr.successors))
    contains(hdr, probabilities) ||
        throw(ArgumentError("transition row is outside the supplied Dirichlet HDR"))
    return SparseCat(copy(hdr.successors), probabilities)
end
