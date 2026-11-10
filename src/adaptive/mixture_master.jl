# Fully corrective acceleration of the saddle loop. All transition rows remain
# HDR-feasible KKT-oracle columns; this small restricted master only mixes them.
# min_lambda r'lambda + sum_z u_z, lambda in simplex,
#     u_z >= sum_l lambda_l payoff[l,z,k] for every continuation k.

function master_epigraph(scores, mu)
    length(scores) == 1 && return (only(scores), [1.0], zeros(1,1))
    peak = maximum(scores)
    gaps = peak .- scores
    # Eliminate u analytically: sum_k mu/(u-score_k)=1. Solve for its strictly
    # positive distance above the largest score to avoid cancellation.
    low, high = mu, mu*length(scores)
    for _ in 1:48
        midpoint = (low+high)/2
        if sum(mu ./ (gaps .+ midpoint)) > 1
            low = midpoint
        else
            high = midpoint
        end
    end
    distances = gaps .+ (low+high)/2
    beta = mu ./ distances
    beta ./= sum(beta)
    curvature = mu ./ distances.^2
    hessian = Matrix(Diagonal(curvature)) .- (curvature * transpose(curvature)) ./ sum(curvature)
    objective = peak + (low+high)/2 - mu*sum(log, distances)
    return objective, beta, hessian
end

function master_barrier(lambda, rewards, payoff, mu; derivatives=true)
    objective = sum(rewards .* lambda) - mu*sum(log,lambda)
    gradient = rewards .- mu ./ lambda
    hessian = Matrix(Diagonal(mu ./ lambda.^2))
    mixtures = Vector{Float64}[]
    for matrix in payoff
        scores = vec(transpose(matrix)*lambda)
        term, beta, curvature = master_epigraph(scores,mu)
        objective += term
        push!(mixtures,beta)
        if derivatives
            gradient .+= matrix * beta
            hessian .+= matrix * curvature * transpose(matrix)
        end
    end
    return (; objective, gradient, hessian, mixtures)
end

function correct_saddle_mixtures(prepared, weights, columns, scores, gamma, tolerance, warm)
    masses = [planning_edge_mass(prepared,weights,theta) for theta in columns]
    rewards = [sum(mass .* prepared.rewards) for mass in masses]
    payoff = [reduce(vcat,[transpose(gamma .* vec(transpose(matrix)*mass)) for mass in masses])
              for matrix in scores]
    count = length(columns)
    if count == 1
        mixtures = [begin
            beta = zeros(size(matrix,2)); beta[argmax(vec(matrix))] = 1; beta
        end for matrix in payoff]
        return (; theta=deepcopy(only(columns)), mixtures, weights=[1.0])
    end
    scale = max(1.0,maximum(abs,rewards),maximum(maximum(abs,matrix) for matrix in payoff))
    scaled_rewards, scaled_payoff = rewards ./ scale, [matrix ./ scale for matrix in payoff]
    lambda = if warm !== nothing && length(warm) == count-1
        vcat(0.95 .* warm,0.05)
    elseif warm !== nothing && length(warm) == count
        copy(warm)
    else
        fill(1/count,count)
    end
    lambda = max.(lambda,1e-12); lambda ./= sum(lambda)
    constraints = count + sum(size(matrix,2) for matrix in payoff)
    target_mu = max(1e-14, tolerance/(10scale*constraints))
    mu = 0.1
    while true
        for _ in 1:45
            state = master_barrier(lambda,scaled_rewards,scaled_payoff,mu)
            q = count-1
            gradient = state.gradient[1:q] .- state.gradient[end]
            last_column = state.hessian[1:q,end]
            hessian = state.hessian[1:q,1:q] .- last_column .- transpose(last_column) .+ state.hessian[end,end]
            direction = try
                -(cholesky(Symmetric(hessian)) \ gradient)
            catch e
                e isa PosDefException || rethrow()
                regularization = max(1e-14,maximum(abs,hessian)*1e-12)
                -((hessian + regularization*I) \ gradient)
            end
            decrement = -sum(gradient .* direction)
            isfinite(decrement) && decrement > min(1e-12,mu*1e-3) || break
            full_direction = vcat(direction,-sum(direction))
            step = min(1.0, minimum((-0.99lambda[i]/full_direction[i] for i in eachindex(lambda)
                                     if full_direction[i] < 0); init=1.0))
            accepted = false
            for _ in 1:40
                next = lambda .+ step .* full_direction
                if all(>(0),next)
                    next ./= sum(next)
                    next_state = master_barrier(next,scaled_rewards,scaled_payoff,mu; derivatives=false)
                    if next_state.objective <= state.objective - 1e-4step*decrement
                        lambda = next
                        accepted = true
                        break
                    end
                end
                step /= 2
            end
            accepted || break
        end
        mu == target_mu && break
        mu = max(target_mu,0.2mu)
    end
    final = master_barrier(lambda,scaled_rewards,scaled_payoff,mu)
    theta = [sum((lambda[l] .* columns[l][i] for l in eachindex(columns)))
             for i in eachindex(prepared.parents)]
    return (; theta, mixtures=final.mixtures, weights=lambda)
end
