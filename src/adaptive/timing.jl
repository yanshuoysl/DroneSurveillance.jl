# Wall-clock phase measurements; Julia's compile_time includes recompile_time.
# Warm up the same workload first, and report residual compilation separately.
function computation_phase_timing(stats)
    compilation = hasproperty(stats, :compile_time) ? stats.compile_time : nothing
    return (; wall_seconds=stats.time, compilation_seconds=compilation,
        execution_seconds=compilation === nothing ? nothing : max(0.0, stats.time-compilation),
        gc_seconds=stats.gctime)
end

function computation_timing_summary(steps, episode_wall_seconds)
    total(f) = sum(f, steps; init=0.0)
    phases = (:planning, :environment, :belief_update)
    compilation_known = all(t -> all(p -> t[p].compilation_seconds !== nothing, phases), steps)
    wall_seconds = total(t -> sum(t[p].wall_seconds for p in phases))
    compilation_seconds = compilation_known ?
        total(t -> sum(t[p].compilation_seconds for p in phases)) : nothing
    computation_seconds = compilation_known ?
        total(t -> sum(t[p].execution_seconds for p in phases)) : nothing
    return (; steps=length(steps), computation_seconds, core_wall_seconds=wall_seconds,
        compilation_seconds, episode_wall_seconds,
        planning_seconds=total(t -> t.planning.wall_seconds),
        environment_seconds=total(t -> t.environment.wall_seconds),
        belief_update_seconds=total(t -> t.belief_update.wall_seconds),
        gc_seconds=total(t -> sum(t[p].gc_seconds for p in phases)),
        includes_gc=true, excludes=(:logging, :plotting, :gif, :step_images, :startup, :warmup),
        method=:phase_wall_minus_reported_compilation)
end

function log_computation_timing!(logger, timing, step)
    episode_event(logger, :step_timing; step, timing)
    if logger !== nothing && logger.terminal !== nothing
        seconds = sum(p.wall_seconds for p in (timing.planning,timing.environment,timing.belief_update))
        println(logger.terminal,"  computation wall time: ",round(seconds;digits=6),
            " s (planning=",round(timing.planning.wall_seconds;digits=6),
            ", environment=",round(timing.environment.wall_seconds;digits=6),
            ", belief update=",round(timing.belief_update.wall_seconds;digits=6),")")
    end
end
