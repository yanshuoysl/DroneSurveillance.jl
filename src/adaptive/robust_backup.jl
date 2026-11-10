# Paper layout (time index n, terminal layer N):
# V_n(b) = max_a min_Theta sum_s b(s) [r_Theta(s,a)
#                 + gamma * sum_(t,z) p_Theta(t,z|s,a) V_(n+1)(b')].
# r_Theta=sum_t T_Theta*R(s,a,t); p_Theta=T_Theta*O. The same Theta is
# used in both terms and for every observation, before the real observation.

struct RobustActionBackup
    action::Int
    stage::Int
    value::Float64
    lower_bound::Float64
    upper_bound::Float64
    gap::Float64
    current_reward::Float64
    expected_future_value::Float64
    selected_rows::BeliefTransitionRows
    continuation::RobustContinuation
    iterations::Int
    status::Symbol
end

struct RobustDecision
    action::Union{Nothing,Int}
    stage::Int
    value::Float64
    lower_bound::Float64
    upper_bound::Float64
    selection_gap::Float64
    selected_rows::Union{Nothing,BeliefTransitionRows}
    action_results::Vector{RobustActionBackup}
    status::Symbol
end

Base.show(io::IO, r::RobustActionBackup) = print(io,"RobustActionBackup(action=",r.action,
    ", stage=",r.stage,", bounds=(",r.lower_bound,", ",r.upper_bound,"), gap=",r.gap,
    ", iterations=",r.iterations,")")
Base.show(io::IO, d::RobustDecision) = print(io,"RobustDecision(action=",d.action,
    ", stage=",d.stage,", value=",d.value,", selection_gap=",d.selection_gap,", status=",d.status,")")
Base.show(io::IO, ::MIME"text/plain", r::RobustActionBackup) = show(io,r)
Base.show(io::IO, ::MIME"text/plain", d::RobustDecision) = show(io,d)

function continuation_scores(vf, prepared, n)
    families = [copy(continuation_family!(vf, states, n+1)) for states in prepared.next_supports]
    entries = sum(length(prepared.children) * length(family) for family in families; init=0)
    entries <= vf.options.max_score_entries ||
        throw(RobustPlanningError(:resource, "backup exceeds max_score_entries"))
    scores = Matrix{Float64}[]
    for (z, family) in enumerate(families)
        positions = Dict(s => j for (j,s) in enumerate(prepared.next_supports[z]))
        matrix = zeros(length(prepared.children), length(family))
        for e in eachindex(prepared.children)
            isfinite(prepared.log_likelihoods[e,z]) || continue
            probability = exp(prepared.log_likelihoods[e,z])
            j = positions[prepared.children[e]]
            for k in eachindex(family)
                matrix[e,k] = probability * family[k].coefficients[j]
            end
        end
        push!(scores, matrix)
    end
    return scores
end

function planning_edge_mass(prepared, weights, theta)
    mass = zeros(length(prepared.children))
    for i in eachindex(weights)
        mass[prepared.ranges[i]] .= weights[i] .* theta[i]
    end
    return mass
end

function paper_objective(prepared, weights, theta, scores, gamma)
    mass = planning_edge_mass(prepared, weights, theta)
    # Current reward and joint transition-observation continuation term.
    r_theta = sum(mass .* prepared.rewards)
    future = sum(maximum(vec(transpose(matrix) * mass)) for matrix in scores; init=0.0)
    return (; value=r_theta + gamma * future, current_reward=r_theta, future)
end

function planning_row_oracle(vf, prepared, weights, mixtures, scores, gamma)
    coefficients = copy(prepared.rewards)
    for (matrix, mixture) in zip(scores, mixtures)
        coefficients .+= gamma .* (matrix * mixture)
    end
    theta = Vector{Float64}[]
    row_lower_bounds = Float64[]
    for i in eachindex(prepared.parents)
        row_values = coefficients[prepared.ranges[i]]
        result = planner_row_minimum(vf, prepared, i, row_values)
        push!(theta, result === nothing ? [1.0] : copy(result.theta))
        push!(row_lower_bounds, result === nothing ? only(row_values) : result.objective_lower_bound)
    end
    return (; theta, row_lower_bounds, lower_bound=sum(weights .* row_lower_bounds))
end

function project_planning_simplex(values)
    length(values) == 1 && return [1.0]
    ordered = sort(values; rev=true)
    total = 0.0
    threshold = 0.0
    for j in eachindex(ordered)
        total += ordered[j]
        candidate = (total - 1) / j
        if ordered[j] > candidate
            threshold = candidate
        end
    end
    projected = max.(values .- threshold, 0.0)
    isfinite(sum(projected)) && sum(projected) > 0 ||
        throw(RobustPlanningError(:numerical, "continuation mixture projection failed"))
    return projected ./ sum(projected)
end

function improve_primal_rows(prepared, weights, current, candidate, scores, gamma)
    # Exact one-dimensional convex piecewise-linear search along a feasible
    # segment. This accelerates primal averaging without another row solve.
    mass = planning_edge_mass(prepared, weights, current)
    delta_mass = planning_edge_mass(prepared, weights, candidate) .- mass
    reward0 = sum(mass .* prepared.rewards)
    reward_slope = sum(delta_mass .* prepared.rewards)
    offsets = [vec(transpose(matrix) * mass) for matrix in scores]
    slopes = [vec(transpose(matrix) * delta_mass) for matrix in scores]
    function along(t)
        objective = reward0 + t * reward_slope
        left, right = reward_slope, reward_slope
        for (offset, slope) in zip(offsets, slopes)
            values = offset .+ t .* slope
            maximum_value = maximum(values)
            active = findall(==(maximum_value), values)
            objective += gamma * maximum_value
            left += gamma * minimum(slope[active])
            right += gamma * maximum(slope[active])
        end
        return objective, left, right
    end
    f0, _, right0 = along(0.0)
    f1, left1, _ = along(1.0)
    best_t, best_value = f1 < f0 ? (1.0, f1) : (0.0, f0)
    if right0 < 0 && left1 > 0
        low, high = 0.0, 1.0
        for _ in 1:55
            midpoint = (low + high) / 2
            value, left, right = along(midpoint)
            if value < best_value
                best_t, best_value = midpoint, value
            end
            if left <= 0 <= right
                break
            elseif right < 0
                low = midpoint
            else
                high = midpoint
            end
        end
    end
    return [(1-best_t) .* x .+ best_t .* y for (x,y) in zip(current,candidate)]
end

function finish_action_backup(vf, b, prepared, n, theta, scores, lower_bound,
                              lower_coefficients, iterations, tolerance)
    rows = Dict{AdaptiveState,SelectedTransitionRow}()
    normalized = Vector{Float64}[]
    for i in eachindex(prepared.parents)
        hdr = prepared.hdrs[i]
        if hdr === nothing
            push!(normalized, [1.0])
        else
            selected = SelectedTransitionRow(theta[i], hdr)
            rows[prepared.parents[i]] = selected
            push!(normalized, collect(selected.theta))
        end
    end
    selected_rows = BeliefTransitionRows(vf.pomdp, b, prepared.action, rows)
    objective = paper_objective(prepared, planning_weights(b,prepared.support), normalized,
                                scores, discount(vf.pomdp))
    gap = objective.value - lower_bound
    gap >= -max(1e-12,64eps()*max(1.0,abs(objective.value))) ||
        throw(RobustPlanningError(:numerical, "primal action value fell below its lower bound"))
    max(0.0,gap) <= tolerance || return nothing
    continuation = RobustContinuation(Int(n), prepared.support, Tuple(lower_coefficients), prepared.action)
    return RobustActionBackup(prepared.action, Int(n), lower_bound, lower_bound,
        max(objective.value,lower_bound), max(0.0,gap), objective.current_reward, objective.future,
        selected_rows, continuation, iterations, :converged)
end

"""
    backup_action(pomdp, belief, action, n, value_function)

Solve the paper-layout inner minimum for one available action with fixed next
continuation collections and HDRs. Dual mixtures choose observation-specific
continuations; `minimize_hdr_row` is the linear row oracle. Projected dual
supergradient ascent and feasible primal row averaging are checked by a
primal-dual gap, including each row oracle's error bound. Failures throw.

Bounds apply to the specified finite continuation approximation and calibrated
HDRs, not to the unknown true model or unrepresented continuations. The returned
`value` is the lower bound; `selected_rows` is the feasible primal assignment.
Count increments occur in latent child construction only, never in the parents.
Hypothetical observations do not prepare or consume the real belief updater.
"""
function backup_action(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief,
                       action::Integer, n::Integer, vf::RobustValueFunction)
    pomdp === vf.pomdp || throw(ArgumentError("value function belongs to a different model"))
    check_planning_stage(vf,n)
    n < vf.horizon || throw(ArgumentError("no action backup at the terminal horizon"))
    states = planning_support(vf,b)
    require_belief_action(pomdp,b,action)
    all(s -> s.outcome != ACTIVE_STATE, states) &&
        throw(ArgumentError("terminal-only beliefs have no action backup"))
    final_layer = n == vf.horizon-1
    prepared = prepare_robust_action!(vf,states,action;need_observations=!final_layer)
    weights = planning_weights(b,states)
    if final_layer
        # With V_N=0, rectangular row minima directly give the whole backup.
        rows = Dict{AdaptiveState,SelectedTransitionRow}()
        lower_coefficients = Float64[]
        upper_coefficients = Float64[]
        for i in eachindex(states)
            result = planner_row_minimum(vf,prepared,i,prepared.rewards[prepared.ranges[i]])
            if result === nothing
                r = only(prepared.rewards[prepared.ranges[i]])
                push!(lower_coefficients,r); push!(upper_coefficients,r)
            else
                rows[prepared.parents[i]] = SelectedTransitionRow(result.theta,prepared.hdrs[i])
                push!(lower_coefficients,result.objective_lower_bound)
                push!(upper_coefficients,result.value)
            end
        end
        lower = sum(weights .* lower_coefficients)
        upper = max(lower,sum(weights .* upper_coefficients))
        gap = upper-lower
        gap <= vf.options.gap_tolerance/2 || throw(RobustPlanningError(:convergence,"one-step row gap is too large"))
        selected = BeliefTransitionRows(pomdp,b,action,rows)
        continuation = RobustContinuation(Int(n),states,Tuple(lower_coefficients),Int(action))
        return RobustActionBackup(Int(action),Int(n),lower,lower,upper,gap,
            sum(weights .* upper_coefficients),0.0,selected,continuation,1,:converged)
    end
    scores = continuation_scores(vf,prepared,n)
    gamma = discount(pomdp)
    mixtures = [fill(1 / size(matrix,2),size(matrix,2)) for matrix in scores]
    best_lower = -Inf
    best_coefficients = zeros(length(states))
    best_upper = Inf
    best_theta = Vector{Float64}[]
    averaged = nothing
    columns = Vector{Vector{Float64}}[]
    master_weights = nothing
    tolerance = vf.options.gap_tolerance / 2
    for iteration in 1:vf.options.max_iterations
        oracle = planning_row_oracle(vf,prepared,weights,mixtures,scores,gamma)
        # Retain only distinct feasible oracle assignments for the restricted
        # mixture master. Numerical equality here affects acceleration only.
        if !any(c -> all(isapprox(x,y;atol=1e-13,rtol=1e-13) for (x,y) in zip(c,oracle.theta)),columns)
            length(columns) < vf.options.max_saddle_columns ||
                throw(RobustPlanningError(:resource,"max_saddle_columns reached before gap convergence"))
            push!(columns,deepcopy(oracle.theta))
        end
        if oracle.lower_bound > best_lower
            best_lower = oracle.lower_bound
            best_coefficients = copy(oracle.row_lower_bounds)
        end
        averaged = averaged === nothing ? deepcopy(oracle.theta) :
            improve_primal_rows(prepared,weights,averaged,oracle.theta,scores,gamma)
        for candidate in (oracle.theta, averaged)
            upper = paper_objective(prepared,weights,candidate,scores,gamma).value
            if upper < best_upper
                best_upper, best_theta = upper, deepcopy(candidate)
            end
        end
        if best_upper - best_lower <= tolerance
            result = finish_action_backup(vf,b,prepared,n,best_theta,scores,best_lower,
                                          best_coefficients,iteration,tolerance)
            result === nothing || return result
        end
        mass = planning_edge_mass(prepared,weights,oracle.theta)
        gradients = [gamma .* vec(transpose(matrix)*mass) for matrix in scores]
        centered = [g .- sum(g)/length(g) for g in gradients]
        norm2 = sum(sum(abs2,g) for g in centered; init=0.0)
        norm2 > 0 && isfinite(norm2) ||
            throw(RobustPlanningError(:convergence, "dual gradient vanished before the backup gap closed (oracle/arithmetic precision limit)"))
        step = vf.options.dual_step_scale * max(0.0,best_upper-oracle.lower_bound) / norm2
        isfinite(step) || throw(RobustPlanningError(:numerical,"dual step exceeds numerical range"))
        next = [project_planning_simplex(beta .+ step .* g) for (beta,g) in zip(mixtures,centered)]
        next == mixtures && throw(RobustPlanningError(:convergence,"continuation mixtures stalled before verification"))
        if iteration >= 4
            correction = correct_saddle_mixtures(prepared,weights,columns,scores,gamma,tolerance,master_weights)
            master_weights = correction.weights
            corrected = paper_objective(prepared,weights,correction.theta,scores,gamma).value
            if corrected < best_upper
                best_upper, best_theta = corrected, deepcopy(correction.theta)
            end
            averaged = correction.theta
            mixtures = correction.mixtures
        else
            mixtures = next
        end
    end
    throw(RobustPlanningError(:convergence,
        "action $action at stage $n did not converge; inner gap=$(best_upper-best_lower)"))
end

"""Compare all six actions and return the best lower-bound action.
Selection gaps concern the same finite continuation approximation as the action
backups. Horizon and terminal decisions return action=nothing and value zero.
With tied_action_rdn=true, sample uniformly among numerical maxima using rng;
otherwise retain the first action in ID order. Unique maxima consume no RNG.
"""
function robust_decision(pomdp::DroneSurveillancePOMDP, b::ExactAdaptiveBelief,
                         n::Integer, vf::RobustValueFunction;
                         tied_action_rdn::Bool=false,rng::AbstractRNG=Random.default_rng())
    pomdp === vf.pomdp || throw(ArgumentError("value function belongs to a different model"))
    check_planning_stage(vf,n)
    states = planning_support(vf,b)
    if n == vf.horizon || all(s -> s.outcome != ACTIVE_STATE, states)
        return RobustDecision(nothing,Int(n),0.0,0.0,0.0,0.0,nothing,
            RobustActionBackup[],n == vf.horizon ? :horizon : :terminal)
    end
    results = RobustActionBackup[backup_action(pomdp,b,a,n,vf) for a in actions(pomdp,b)]
    isempty(results) && throw(RobustPlanningError(:actions,"no action is available throughout the belief support"))
    best_lower = maximum(r.lower_bound for r in results)
    tie_tolerance = min(vf.options.gap_tolerance/2,64eps()*max(1.0,abs(best_lower)))
    tied = filter(r -> r.lower_bound >= best_lower-tie_tolerance,results)
    winner = tied_action_rdn && length(tied) > 1 ? rand(rng,tied) : first(tied)
    upper = maximum(r.upper_bound for r in results)
    gap = max(0.0,upper-winner.lower_bound)
    gap <= vf.options.gap_tolerance ||
        throw(RobustPlanningError(:selection,"action comparison gap is too large: $gap"))
    return RobustDecision(winner.action,Int(n),winner.lower_bound,winner.lower_bound,
                          upper,gap,winner.selected_rows,results,:converged)
end

"""Prepare the converged decision's model before executing its real action."""
function prepare_decision!(updater::ExactAdaptiveUpdater, b::ExactAdaptiveBelief,
                           decision::RobustDecision)
    decision.status == :converged && decision.action !== nothing ||
        throw(ArgumentError("cannot prepare a horizon/terminal decision"))
    prepare_belief_update!(updater,b,decision.action,decision.selected_rows)
    return updater
end
