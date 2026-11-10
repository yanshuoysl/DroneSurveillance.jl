"""
    configuration_stratified_particles(belief, N; rng)

Compress a weighted hyperstate distribution to at most N retained histories.
Strata are complete configurations (position, heading, outcome). Every stratum
gets at least one draw; remaining draws are allocated by stratum probability.
Within each stratum, draw mode/count histories conditionally, with replacement,
and assign total weight equal to the stratum's original mass. Thus configuration
marginals and configuration support are preserved exactly for this input
while mode/count-history distributions are approximated. No fractional count
increments are used: every retained history carries its own full phi snapshot.

Uses ExactAdaptiveBelief as a finite weighted distribution container, not as a
claim that the resampled posterior is the exact posterior. If its support already
fits N, no sampling is necessary. This function never reads simulator truth.
"""
function configuration_stratified_particles(b::ExactAdaptiveBelief,N::Integer;
        rng::AbstractRNG=Random.default_rng())
    N > 0 || throw(ArgumentError("particle count must be positive"))
    groups = Dict{Tuple{DSState,AdaptiveOutcome},Vector{BeliefHyperstate}}()
    for key in sort!(collect(keys(b._logprobs));by=planning_key_order)
        push!(get!(groups,(key.config,key.outcome),BeliefHyperstate[]),key)
    end
    length(groups) <= N || throw(ArgumentError("particle budget cannot preserve all configurations"))
    if length(b) <= N
        return (;belief=b,expanded=length(b),retained=length(b),configurations=length(groups),resampled=false)
    end
    ordered = sort!(collect(keys(groups));by=g -> (Tuple(g[1].quad)...,g[1].heading,Int(g[2])))
    logmass = [foldl(belief_logadd,(b._logprobs[k] for k in groups[g]);init=-Inf) for g in ordered]
    probabilities = exp.(logmass .- maximum(logmass))
    probabilities ./= sum(probabilities)
    allocations = ones(Int,length(ordered))
    stratum_distribution = Distributions.Categorical(probabilities)
    for _ in 1:N-length(ordered)
        allocations[rand(rng,stratum_distribution)] += 1
    end
    masses = Dict{BeliefHyperstate,Float64}()
    for (i,g) in enumerate(ordered)
        members = groups[g]
        logs = [b._logprobs[k] for k in members]
        conditional = exp.(logs .- maximum(logs))
        conditional ./= sum(conditional)
        distribution = Distributions.Categorical(conditional)
        weight = logmass[i]-log(allocations[i])
        for _ in 1:allocations[i]
            key = members[rand(rng,distribution)]
            masses[key] = belief_logadd(get(masses,key,-Inf),weight)
        end
    end
    posterior = ExactAdaptiveBelief(masses,Val(:logmass))
    return (;belief=posterior,expanded=length(b),retained=length(posterior),
        configurations=length(groups),resampled=true)
end
