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
