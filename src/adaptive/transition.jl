# Combined kernels over (r, m, phi). The separate AR-POMDP factors live in
# locomotion.jl and parameter_update.jl; this file composes their probabilities.

"""
    posterior_predictive_row(pomdp, s::AdaptiveState, action)

Return `(successors, counts, probabilities)` for a nonterminal hyperstate.
Successors follow the full Cartesian support S(r,m,a); probabilities are
`counts / sum(counts)`, the Dirichlet posterior predictive row. Each joint
configuration/mode pair has its own parameter; no factorization is imposed.

Only this (r,m,a) row of `s.counts` is read, with an implicit prior count of one
per feasible pair. Terminal states have no learned row and raise ArgumentError.
This function does not increment counts or sample a latent transition model.
It is a Bayesian baseline, not the AR-POMDP's uncertain locomotion/mode row.
"""
function posterior_predictive_row(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                                  action::Integer)
    transition_source(pomdp, s, action)
    successors = feasible_successors(pomdp, s.config, s.mode, action)
    counts = count_row(pomdp, s.counts, s.config, s.mode, action, successors)
    # Sum in floating point so a row of large, individually valid Int counts
    # cannot overflow the integer accumulator.
    alpha = Float64.(counts)
    probabilities = alpha ./ sum(alpha)
    return (; successors, counts, probabilities)
end

function transition_distribution(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                                 action::Integer, successors, probabilities)
    next_states = AdaptiveState[]
    for (config, mode) in successors
        parameters = parameter_update_transition(pomdp, s, action, config, mode)
        counts = parameters.val # the conditional distribution has exactly one outcome
        push!(next_states, adaptive_state(pomdp, config, mode, counts))
    end
    return SparseCat(next_states, probabilities)
end

"""
    adaptive_transition(pomdp, s::AdaptiveState, action)
    adaptive_transition(pomdp, s::AdaptiveState, action, theta)

Return a distribution over next hyperstates `(r',m',phi')`. On each feasible
branch, phi' increments exactly the count indexed by (r,m,a,r',m'). All other
counts remain unchanged. Each branch owns a separate count snapshot, including
stay branches and branches entering a goal or unsafe state.

Without `theta`, use the Dirichlet posterior predictive row. An explicit `theta`
must be a probability vector in the order of `feasible_successors`; this permits
a fixed physical model, a posterior sample, or a candidate row. It is checked
for simplex membership only. Use `robust_transition` to also require HDR
membership. There is no deterministic motion rule based on a fault flag.

Goal/unsafe sources are absorbing and keep their mode and counts. An explicit
row for such a source must be `[1]`. Invalid source labels or actions are rejected.
These kernels condition on a hidden-state hypothesis; they do not expose the
true mode to the robot or perform the robot's observation-based belief update.
The existing DSState POMDP interface is separate from these augmented kernels.
"""
function adaptive_transition(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                             action::Integer)
    source = transition_source(pomdp, s, action)
    is_terminal(source) && return Deterministic(s)
    row = posterior_predictive_row(pomdp, s, action)
    return transition_distribution(pomdp, s, action, row.successors, row.probabilities)
end

function adaptive_transition(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                             action::Integer, theta::AbstractVector{<:Real})
    source = transition_source(pomdp, s, action)
    successors = feasible_successors(pomdp, s.config, s.mode, action)
    probabilities = transition_probabilities(theta, length(successors))
    is_terminal(source) && return Deterministic(s)
    return transition_distribution(pomdp, s, action, successors, probabilities)
end

"""
    robust_transition(pomdp, s::AdaptiveState, action, theta; hdr::DirichletHDR)

Instantiate an AR-POMDP transition kernel using a row `theta` chosen from a
previously calibrated HDR. `theta` is ordered as `hdr.successors`; a complete
reordering of S(r,m,a) is allowed. The HDR must match both the complete support
and this hyperstate's current row counts. Incomplete, unrelated, or stale HDRs
are rejected, as are vectors outside the simplex or the HDR.

This validates and applies a candidate row; it does not choose a worst-case row.
That choice requires the robust planner's continuation values. The calibrated
HDR retains its existing Monte Carlo approximation; no threshold is reestimated
here. All count increments are the same as in `adaptive_transition`.

Terminal sources have no HDR row: use their absorbing `adaptive_transition`.
"""
function robust_transition(pomdp::DroneSurveillancePOMDP, s::AdaptiveState,
                           action::Integer, theta::AbstractVector{<:Real};
                           hdr::DirichletHDR)
    physical = locomotion_mode_transition(pomdp, s, action, theta; hdr)
    return transition_distribution(pomdp, s, action, physical.vals, physical.probs)
end
