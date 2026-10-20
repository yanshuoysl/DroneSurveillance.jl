# Second transition factor: T(phi' | r,r',m,m',phi,a).

"""
    parameter_update_transition(pomdp, s::AdaptiveState, action, next_config, next_mode)
    parameter_update_transition(pomdp, config, mode, counts, action, next_config, next_mode)

Return the deterministic conditional distribution over the full next count
vector phi', given a feasible latent transition (r,m,a,r',m'). For an active
source, its mass is 1 exactly at phi + delta^a_{(r,m),(r',m')} and 0 at every
other count vector. Query this factor with `pdf(distribution, candidate_counts)`.

The delta has one entry equal to 1 and all other entries equal to 0. Only that
joint transition's count is incremented; all other destinations, modes, actions,
and source rows retain their counts. The implicit prior of one is preserved by
incrementing the sparse phi-1 storage entry. Stay transitions and entry into
goal/unsafe outcomes also increment their own count once.

The resulting DirichletCounts owns a new dictionary. Input counts and sibling
histories are not mutated. No theta, HDR, posterior normalization, or observation
is used here: this is conditional on a latent transition hypothesis, not the
robot's observation-based belief update. Impossible conditioning pairs raise
ArgumentError because they are outside the model's transition support.

For an already terminal source, only its absorbing (r,m) destination is allowed;
return a new snapshot of unchanged counts, with no further learned transition.
The AdaptiveState overload additionally checks the source outcome label.
"""
function parameter_update_transition(pomdp::DroneSurveillancePOMDP,
                                      config::DSState, mode::FaultMode,
                                      counts::DirichletCounts, action::Integer,
                                      next_config::DSState, next_mode::FaultMode)
    source = support_source(pomdp, config, mode, action)
    if is_terminal(source)
        is_feasible_successor(pomdp, config, mode, action, next_config, next_mode) ||
            throw(ArgumentError("terminal sources only allow their absorbing configuration and mode"))
        return Deterministic(DirichletCounts(counts.increments))
    end
    key = CountKey(pomdp, config, mode, action, next_config, next_mode)
    updated = increment_count(counts, key)
    return Deterministic(updated)
end

function parameter_update_transition(pomdp::DroneSurveillancePOMDP,
                                      s::AdaptiveState, action::Integer,
                                      next_config::DSState, next_mode::FaultMode)
    transition_source(pomdp, s, action)
    return parameter_update_transition(pomdp, s.config, s.mode, s.counts,
                                        action, next_config, next_mode)
end
