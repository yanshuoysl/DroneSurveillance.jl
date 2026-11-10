# Read-only episode diagnostics. No diagnostic is passed back to the agent.

mutable struct EpisodeLogger
    terminal::Union{Nothing,IO}
    file::Union{Nothing,IO}
    details::Bool
    top_k::Int
    ids::Dict{BeliefHyperstate,Int}
end
EpisodeLogger(terminal, file; details=true, top_k=3) =
    EpisodeLogger(terminal, file, details, top_k, Dict{BeliefHyperstate,Int}())

# Small JSON encoder for our primitive-only log schema, avoiding a new package.
function episode_json(io::IO, x)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Real
        isfinite(x) ? print(io, x) : episode_json(io, string(x))
    elseif x isa AbstractString || x isa Symbol
        print(io, '"')
        for c in string(x)
            if c == '"' || c == '\\'
                print(io, '\\', c)
            elseif Int(c) < 0x20
                print(io, "\\u", string(Int(c); base=16, pad=4))
            else
                print(io, c)
            end
        end
        print(io, '"')
    elseif x isa NamedTuple || x isa AbstractDict
        print(io, '{')
        for (i, (k, v)) in enumerate(pairs(x))
            i > 1 && print(io, ',')
            episode_json(io, string(k)); print(io, ':'); episode_json(io, v)
        end
        print(io, '}')
    elseif x isa Tuple || x isa AbstractVector
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ',')
            episode_json(io, v)
        end
        print(io, ']')
    else
        throw(ArgumentError("unsupported episode log value: $(typeof(x))"))
    end
end

function episode_event(logger, event; kwargs...)
    logger === nothing && return
    logger.file === nothing && return
    episode_json(logger.file, (; event, kwargs...))
    println(logger.file)
    flush(logger.file)
end

configuration_record(r::DSState) = (; position=Tuple(r.quad), heading=r.heading)
count_key_record(k::CountKey) = (; source=configuration_record(k.config),
    mode=k.mode, action=k.action, destination=configuration_record(k.next_config), next_mode=k.next_mode)

function hypothesis_id!(logger, key)
    return get!(logger.ids, key) do
        length(logger.ids) + 1
    end
end

function hypothesis_record(logger, key)
    return (; id=hypothesis_id!(logger, key), configuration=configuration_record(key.config),
        mode=key.mode, outcome=Symbol(string(key.outcome)),
        increments=[(; key=count_key_record(k), increment=n) for (k,n) in key.increments])
end

function belief_record(logger, b::ExactAdaptiveBelief; pomdp=nothing)
    positions = Dict{Tuple,Float64}()
    modes = Dict{FaultMode,Float64}()
    ordered = sort!(collect(keys(b._logprobs)); by=planning_key_order)
    for k in ordered
        hypothesis_id!(logger, k)
        pos = (Tuple(k.config.quad), k.config.heading, string(k.outcome))
        positions[pos] = belief_logadd(get(positions, pos, -Inf), b._logprobs[k])
        modes[k.mode] = belief_logadd(get(modes, k.mode, -Inf), b._logprobs[k])
    end
    top = sort(ordered; by=k -> -b._logprobs[k])[1:min(logger.top_k, length(ordered))]
    return (; size=length(b),
        configurations=[(; position=p[1], heading=p[2], outcome=p[3], probability=exp(w), log_probability=w,
                            available_actions=pomdp === nothing ? nothing :
                                p[3] == "ACTIVE_STATE" ? actions(pomdp, DSState(p[1],p[2])) : Int64[])
                        for (p,w) in sort!(collect(positions); by=first)],
        modes=[(; mode=m, probability=exp(w), log_probability=w)
               for (m,w) in sort!(collect(modes); by=p -> mode_index(first(p)))],
        top=[(; hypothesis_record(logger,k)..., probability=exp(b._logprobs[k]),
                log_probability=b._logprobs[k]) for k in top],
        hypotheses=logger.details ? [(; hypothesis_record(logger,k)...,
            log_probability=b._logprobs[k]) for k in ordered] : nothing)
end
belief_record(logger, b; pomdp=nothing) = nothing

function boundary_action_diagnostics(p, prior)
    prior === nothing && return nothing
    diagnostics = NamedTuple[]
    for a in actions(p)
        blockers = [c for c in prior.configurations
                    if c.outcome == "ACTIVE_STATE" && !command_in_bounds(p,DSState(c.position,c.heading),a)]
        isempty(blockers) && continue
        log_mass = foldl(belief_logadd, (c.log_probability for c in blockers); init=-Inf)
        witness = blockers[argmax([c.log_probability for c in blockers])]
        push!(diagnostics, (;action=a, outward_probability=exp(log_mass),
            outward_log_probability=log_mass,
            witness=(;position=witness.position,heading=witness.heading,
                probability=witness.probability,log_probability=witness.log_probability)))
    end
    return diagnostics
end

named_actions(available) = [(a, action_name(a)) for a in available]

function log_decision!(logger, p, b, a, decision, step;
                       episode_stage=nothing,lookahead=nothing,planning_mode=nothing,fit=nothing)
    logger === nothing && return
    prior = belief_record(logger, b; pomdp=p)
    boundary_risks = boundary_action_diagnostics(p, prior)
    candidates = decision === nothing ? nothing :
        [(; action=r.action, value=r.value, lower_bound=r.lower_bound, upper_bound=r.upper_bound,
             gap=r.gap, current_reward=r.current_reward, future_value=r.expected_future_value,
             iterations=r.iterations) for r in decision.action_results]
    rows = if decision === nothing || !logger.details
        nothing
    else
        [(; parent_id=hypothesis_id!(logger,k), row_id="$(hypothesis_id!(logger,k))/$a",
            counts=row.alpha, probabilities=row.theta, tau=row.tau, epsilon=row.epsilon,
            successors=[(; configuration=configuration_record(r), mode=m) for (r,m) in row.successors])
         for (k,row) in sort!(collect(decision.selected_rows._rows); by=p -> planning_key_order(first(p)))]
    end
    episode_event(logger, :decision; step, stage=decision === nothing ? nothing : decision.stage,
        episode_stage,lookahead,planning_mode,planning_fit=fit,
        action=a, available_actions=b isa ExactAdaptiveBelief ? actions(p,b) : nothing,
        selection_gap=decision === nothing ? nothing : decision.selection_gap,
        candidates, selected_rows=rows, prior, boundary_risks)
    if logger.terminal !== nothing
        io = logger.terminal
        println(io, "Step ", step, ": action=", a, " (", action_name(a), ")",
            decision === nothing ? "" : "; episode_stage=$episode_stage, planning_stage=$(decision.stage), L=$lookahead, selection_gap=$(decision.selection_gap)")
        fit === nothing || println(io,"  local fit: ",fit.points," training beliefs; ",fit.updates," value updates")
        candidates === nothing || println(io, "  action values: ", [(r.action, round(r.value; digits=6)) for r in candidates])
        prior === nothing || println(io, "  belief before: ", prior.size, " hypotheses")
        if prior !== nothing
            println(io, "  belief available actions: ", named_actions(actions(p,b)))
            println(io, "  belief poses BEFORE action (position, heading in degrees, probability, available actions):")
            for c in prior.configurations
                println(io, "    position=",c.position,"; heading=",c.heading,
                    "; probability=",c.probability,"; log_probability=",c.log_probability,
                    "; available actions=",named_actions(c.available_actions))
            end
            for block in boundary_risks
                println(io,"  boundary risk: ",block.action," (",action_name(block.action),
                    "); outward probability=",block.outward_probability,
                    "; witness position=",block.witness.position,"; heading=",block.witness.heading,
                    "; witness probability=",block.witness.probability)
            end
        end
        if decision !== nothing
            k = first(sort!(collect(keys(decision.selected_rows._rows)); by=planning_key_order))
            row = decision.selected_rows._rows[k]
            println(io, "  representative row h", hypothesis_id!(logger,k), "/", a,
                ": phi=", row.alpha, "; theta=", Tuple(round.(row.theta; digits=4)))
        end
        flush(io)
    end
end

action_name(a) = first(k for (k,v) in ACTIONS_DICT if v == a)

function log_posterior!(logger, p, b, bp, a, o, decision, step;filter_report=nothing)
    logger === nothing && return
    posterior = belief_record(logger, bp; pomdp=p)
    # Reconstruct diagnostic branch IDs/count changes without modifying counts or
    # reselecting rows. Filtering itself is performed exactly once by the updater.
    if logger.details && decision !== nothing
        for k in sort!(collect(keys(b._logprobs)); by=planning_key_order)
            k.outcome == ACTIVE_STATE || continue
            s = thaw_hyperstate(k)
            row = decision.selected_rows._rows[k]
            for (j, (r,m)) in enumerate(row.successors)
                row.theta[j] > 0 || continue
                dest = adaptive_state(p,r,m)
                loglik = belief_camera_loglikelihood(p,r,dest.outcome,o)
                isfinite(loglik) || continue
                key = CountKey(p,s.config,s.mode,a,r,m)
                phi = parameter_update_transition(p,s,a,r,m).val
                child = freeze_hyperstate(adaptive_state(p,r,m,phi))
                episode_event(logger, :belief_branch; step,
                    parent_id=hypothesis_id!(logger,k), child_id=hypothesis_id!(logger,child),
                    row_id="$(hypothesis_id!(logger,k))/$a", successor_index=j,
                    retained=haskey(bp._logprobs,child),
                    count_key=count_key_record(key), count_before=count_value(s.counts,key),
                    count_after=count_value(phi,key), delta=1,
                    log_likelihood=loglik, log_unnormalized_mass=b._logprobs[k]+log(row.theta[j])+loglik)
            end
        end
    end
    episode_event(logger, :belief_update; step, posterior,filter_report)
    if logger.terminal !== nothing && posterior !== nothing
        println(logger.terminal, "  belief after: ", posterior.size, " hypotheses; top=",
            [(h.id, h.configuration.position, h.mode, round(h.probability; digits=4)) for h in posterior.top])
        println(logger.terminal, "  mode probabilities: ", [(m.mode, round(m.probability; digits=4)) for m in posterior.modes])
        filter_report === nothing || println(logger.terminal,"  particle filter: ",filter_report.expanded,
            " -> ",filter_report.retained," retained histories; ",filter_report.configurations," configurations preserved")
        println(logger.terminal, "  phi: +1 per feasible branch's own key; updated HDRs are used at the next decision")
        flush(logger.terminal)
    end
end

"""Save an SVG line chart of immediate recorded reward against executed step.
Uses Compose's vector backend, with no additional plotting/runtime dependencies.
The CSV and curve use identical rewards. Empty and single-step episodes are valid.
"""
function save_reward_chart(path::AbstractString, history)
    endswith(lowercase(path), ".svg") || throw(ArgumentError("reward chart must have an .svg extension"))
    steps = [Int(s.t) for s in history]
    rewards = [Float64(s.r) for s in history]
    length(unique(steps)) == length(steps) && issorted(steps) ||
        throw(ArgumentError("reward history steps must be distinct and increasing"))
    all(isfinite, rewards) || throw(ArgumentError("rewards must be finite"))
    xmax = max(1, isempty(steps) ? 1 : maximum(steps))
    lo, hi = isempty(rewards) ? (-1.0, 1.0) : extrema(rewards)
    padding = max(1.0, 0.1*(hi-lo)); lo -= padding; hi += padding
    plot_x(t) = 0.12 + 0.82*(t-1)/max(1,xmax-1)
    plot_y(r) = 0.80 - 0.64*(r-lo)/(hi-lo)
    parts = Any[compose(context(), rectangle(), fill("white")),
        compose(context(), text(0.5,0.055,"Immediate reward by step",hcenter), fontsize(15pt)),
        compose(context(), text(0.53,0.96,"Step",hcenter), fontsize(12pt)),
        compose(context(), text(0.015,0.11,"Reward"), fontsize(12pt)),
        compose(context(), line([(0.12,0.14),(0.12,0.80),(0.95,0.80)]), stroke("black"), linewidth(0.3mm))]
    for y in range(lo,hi; length=5)
        push!(parts, compose(context(), line([(0.12,plot_y(y)),(0.95,plot_y(y))]), stroke("#dddddd"), linewidth(0.15mm)))
        push!(parts, compose(context(), text(0.105,plot_y(y),string(round(y; digits=1)),hright,vcenter), fontsize(10pt)))
    end
    for t in unique(round.(Int, range(1,xmax; length=min(8,xmax))))
        push!(parts, compose(context(), text(plot_x(t),0.855,string(t),hcenter), fontsize(10pt)))
    end
    if isempty(steps)
        push!(parts, compose(context(), text(0.53,0.47,"No actions executed",hcenter), fontsize(12pt)))
    else
        length(steps) > 1 && push!(parts, compose(context(), line([(plot_x(t),plot_y(r)) for (t,r) in zip(steps,rewards)]),
            stroke("#2166ac"), linewidth(0.65mm)))
        for (t,r) in zip(steps,rewards)
            push!(parts, compose(context(), circle(plot_x(t),plot_y(r),0.004), fill("#2166ac")))
        end
    end
    # Compose draws earlier children on top; reverse to put the background last.
    draw(SVG(path,900px,420px), compose(context(), reverse(parts)...))
    return abspath(path)
end
