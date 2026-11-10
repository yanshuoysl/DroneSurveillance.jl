# Hidden episode fault times. Activating faults is a separate simulator step.

"""
    sample_fault_schedule(H, N_f; rng=Random.default_rng())

Sample the hidden fault activation steps for an episode starting nominal.
First draw K uniformly from the integers 0:min(N_f, H, N_ACTIONS). Then draw
K distinct steps uniformly without replacement from 1:H and return them sorted.
The number of actions is six; each event will freeze one previously healthy
action. At most one fault is scheduled per step. Both H and N_f must be
nonnegative integers; H must fit in Int. H=0 or N_f=0 returns an empty schedule.

For H=100 and N_f=2, K=0, K=1, and K=2 each have probability 1/3. Conditional
on K, every K-element subset of 1:H is equally likely. This is uniform over
the fault count, not over all possible schedules of different sizes.

This samples times only: it neither freezes actions nor changes a hyperstate,
the learner's successor support, or the available action set. A simulator must
keep the schedule hidden, choose a healthy action uniformly at each event, and
activate it before executing that step's command. Events after early termination
are never activated. No physical simulation or belief update occurs here.
"""
function sample_fault_schedule(H::Integer, N_f::Integer;
                               rng::AbstractRNG=Random.default_rng())
    0 <= H <= typemax(Int) ||
        throw(ArgumentError("H must be nonnegative and fit in Int"))
    N_f >= 0 || throw(ArgumentError("N_f must be nonnegative"))
    max_faults = Int(min(N_f, H, N_ACTIONS))
    K = rand(rng, 0:max_faults)
    steps = Int[]
    # K <= 6, so drawing distinct steps avoids allocating a permutation of H.
    while length(steps) < K
        step = rand(rng, 1:Int(H))
        step in steps || push!(steps, step)
    end
    return sort!(steps)
end

"""
Hidden physical episode configuration. Random schedules use the approved uniform
fault-count law. Optional `fault_steps` and `fault_actions` provide controlled
fixtures; actions must be distinct and initially healthy. The schedule and RNG
belong exclusively to the simulator. Construct a fresh environment per episode.
`N_f` caps total frozen actions, including any supplied initial faults.
"""
struct HiddenFaultEnvironment{R<:AbstractRNG}
    H::Int
    N_f::Int
    fault_steps::Tuple{Vararg{Int}}
    fault_actions::Union{Nothing,Tuple{Vararg{Int}}}
    initial_mode::FaultMode
    rng::R
end

function HiddenFaultEnvironment(H::Integer, N_f::Integer;
        fault_steps=nothing, fault_actions=nothing, initial_mode::FaultMode=NOMINAL_MODE,
        schedule_rng::AbstractRNG=Random.default_rng(),
        fault_rng::AbstractRNG=Random.default_rng())
    0 <= H <= typemax(Int) || throw(ArgumentError("H must be nonnegative and fit in Int"))
    0 <= N_f <= typemax(Int) || throw(ArgumentError("N_f must be nonnegative and fit in Int"))
    existing = count(identity, initial_mode)
    existing <= N_f || throw(ArgumentError("initial faults exceed N_f"))
    fault_steps === nothing && fault_actions !== nothing &&
        throw(ArgumentError("explicit fault_actions require fault_steps"))
    remaining = min(N_f-existing, N_ACTIONS-existing)
    steps = fault_steps === nothing ? sample_fault_schedule(H, remaining; rng=schedule_rng) :
        collect(fault_steps)
    all(t -> t isa Integer && 1 <= t <= H, steps) ||
        throw(ArgumentError("fault steps must be integers inside 1:H"))
    issorted(steps) && length(unique(steps)) == length(steps) ||
        throw(ArgumentError("fault steps must be distinct and sorted"))
    length(steps) <= min(N_f-existing, N_ACTIONS-existing) ||
        throw(ArgumentError("schedule exceeds fault budget or healthy actions"))
    chosen = if fault_actions === nothing
        nothing
    else
        ids = collect(fault_actions)
        length(ids) == length(steps) || throw(ArgumentError("one action is needed per fault step"))
        all(a -> a isa Integer && 1 <= a <= N_ACTIONS && !initial_mode[a], ids) ||
            throw(ArgumentError("fault actions must be initially healthy action IDs"))
        length(unique(ids)) == length(ids) || throw(ArgumentError("an action cannot fail twice"))
        Tuple(Int.(ids))
    end
    return HiddenFaultEnvironment(Int(H), Int(N_f), Tuple(Int.(steps)), chosen,
                                  initial_mode, deepcopy(fault_rng))
end

"""
Execute one physical step, applying a scheduled fault BEFORE the command. Frozen
commands stay; healthy commands move/rotate. Executed outward commands enter the
unsafe terminal state just like obstacle collisions; a frozen command stays.
This does not select a learned row or change the learner's Cartesian support.
One joint count update uses the ORIGINAL source mode and final destination mode.
Returned truth diagnostics are for evaluation only, never policy/updater inputs.
"""
function physical_step(pomdp::DroneSurveillancePOMDP, env::HiddenFaultEnvironment,
                       s::AdaptiveState, a::Integer, step::Integer)
    1 <= step <= env.H || throw(ArgumentError("physical step must lie in 1:H"))
    source = transition_source(pomdp, s, a)
    is_terminal(source) && throw(ArgumentError("cannot execute an action after termination"))
    all(i -> !env.initial_mode[i] || s.mode[i], 1:N_ACTIONS) ||
        throw(ArgumentError("physical mode lost an initial fault"))
    count(identity, s.mode) <= env.N_f || throw(ArgumentError("physical mode exceeds N_f"))
    index = findfirst(==(step), env.fault_steps)
    next_mode, event = s.mode, nothing
    if index !== nothing
        healthy = [a for a in 1:N_ACTIONS if !s.mode[a]]
        count(identity, s.mode) < env.N_f && !isempty(healthy) ||
            throw(ArgumentError("no remaining fault budget"))
        failed = env.fault_actions === nothing ? rand(env.rng, healthy) : env.fault_actions[index]
        failed in healthy || throw(ArgumentError("scheduled action is already frozen"))
        next_mode = ntuple(i -> s.mode[i] || i == failed, N_ACTIONS)
        event = (; step=Int(step), action=failed)
    end
    attempted = commanded_config(s.config, a)
    config = is_failed(next_mode, a) ? s.config : attempted
    destination = adaptive_state(pomdp, config, next_mode)
    key = CountKey(pomdp, s.config, s.mode, a, destination.config, next_mode)
    counts = parameter_update_transition(pomdp, s, a, destination.config, next_mode).val
    sp = adaptive_state(pomdp, destination.config, next_mode, counts)
    return (; sp, event, attempted, count_key=key)
end
