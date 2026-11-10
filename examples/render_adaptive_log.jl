# Replay saved physical configurations only; does not rerun planning/filtering.
# julia --project=test examples/render_adaptive_log.jl [episode.jsonl] [drone_steps.gif]
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..")))
using JSON
using POMDPs
using POMDPTools
if !isdefined(@__MODULE__,:DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end
if !isdefined(@__MODULE__, :save_adaptive_artifacts)
    include(joinpath(@__DIR__,"adaptive_artifacts.jl"))
end

"""Find the latest run log; retain compatibility with older flat output files."""
function latest_adaptive_episode_log(output_root=normpath(joinpath(@__DIR__,"..","test","output")))
    isdir(output_root) || throw(ArgumentError("output directory does not exist: $output_root"))
    logs = [joinpath(dir,"episode.jsonl") for dir in readdir(output_root;join=true)
            if isdir(dir) && isfile(joinpath(dir,"episode.jsonl"))]
    if !isempty(logs)
        return last(sort!(logs;by=path -> (mtime(path),path)))
    end
    legacy_log = joinpath(output_root,"episode.jsonl")
    isfile(legacy_log) || throw(ArgumentError("no saved episode found under $output_root"))
    return legacy_log
end

function saved_action_value_decisions(events)
    return [(;t=Int(e["step"]),action=Int(e["action"]),
        action_values=[(;action=Int(c["action"]),value=Float64(c["value"])) for c in e["candidates"]])
        for e in events if get(e,"event",nothing) == "decision"]
end

"""Add Q-value GIF/JPGs to an existing episode without rerunning the simulation."""
function render_saved_action_value_artifacts(log_path::AbstractString;
        filename=joinpath(dirname(abspath(log_path)),"action_values.gif"),
        jpg_directory=joinpath(dirname(abspath(log_path)),"action_values"),
        generate_gif::Bool=true,generate_jpgs::Bool=true,fps::Integer=2)
    # Ignore large belief-branch records; only decision events contain plotted data.
    events = [JSON.parse(line) for line in eachline(log_path)
              if occursin("\"event\":\"decision\"",line)]
    artifacts = save_action_value_artifacts(saved_action_value_decisions(events);
        filename,jpg_directory,generate_gif,generate_jpgs,fps)
    artifacts.gif_path === nothing || println("Saved action-value animation: ",artifacts.gif_path)
    artifacts.jpg_directory === nothing || println("Saved ",length(artifacts.jpg_paths),
        " action-value JPGs to ",artifacts.jpg_directory)
    return artifacts
end

"""Render combined poses, decision values and faults, including the final destination.
This reconstructs visualization frames, not beliefs or learned count histories.
Existing log/reward files are read-only; an existing GIF is not overwritten.
"""
function render_saved_adaptive_artifacts(log_path::AbstractString;
        filename=joinpath(dirname(abspath(log_path)),"drone_steps.gif"),
        jpg_directory=joinpath(dirname(abspath(log_path)),"steps"),
        generate_gif::Bool=true,generate_jpgs::Bool=true,fps::Integer=2)
    fps > 0 || throw(ArgumentError("fps must be positive"))
    events = [JSON.parse(line) for line in eachline(log_path) if !isempty(strip(line))]
    configs = filter(e -> get(e,"event",nothing) == "configuration",events)
    starts = filter(e -> get(e,"event",nothing) == "start",events)
    stops = filter(e -> get(e,"event",nothing) == "stop",events)
    length(configs) == length(starts) == length(stops) == 1 ||
        throw(ArgumentError("expected one completed episode in the log"))
    transitions = filter(e -> get(e,"event",nothing) == "physical_transition",events)
    length(transitions) == only(stops)["steps"] || throw(ArgumentError("incomplete physical history"))
    decisions = saved_action_value_decisions(events)
    length(decisions) == length(transitions) || throw(ArgumentError("incomplete decision history"))
    DS = DroneSurveillanceAdaptive
    map = only(configs)["map"]
    truth = only(starts)["simulator_truth"]
    camera = reduce(vcat,[permutedims(Float64.(row)) for row in map["camera"]])
    model = DS.DroneSurveillancePOMDP(size=Tuple(Int.(map["size"])),
        region_A=DS.DSPos(map["start"]),region_B=DS.DSPos(map["goal"]),
        initial_heading=Int(truth["state"]["heading"]),
        obstacles=[DS.DSPos(pos) for pos in map["obstacles"]],camera=DS.KnownCamera(camera))
    pose(data,mode) = DS.adaptive_state(model,DS.DSState(data["position"],Int(data["heading"])),
        Tuple(Bool.(mode)))
    current = pose(truth["state"],truth["mode"])
    initial_state = current
    frames = NamedTuple[]
    for (t,e) in enumerate(transitions)
        e["step"] == t || throw(ArgumentError("physical steps must be consecutive"))
        physical = e["simulator_truth"]
        source = pose(physical["source"],physical["source_mode"])
        source == current || throw(ArgumentError("physical poses are discontinuous"))
        destination = pose(physical["destination"],physical["next_mode"])
        decision = decisions[t]
        decision.t == t || throw(ArgumentError("decision steps must be consecutive"))
        decision.action == Int(physical["count_key"]["action"]) ||
            throw(ArgumentError("decision action differs from physical action"))
        push!(frames,(;t,s=source,sp=destination,o=DS.DSObservation(e["observation"]),
            a=decision.action,r=Float64(e["reward"]),decision))
        current = destination
    end
    history = POMDPTools.SimHistory(frames,discount(model),nothing,nothing)
    artifacts = save_adaptive_artifacts(model,history;filename,jpg_directory,initial_state,
        generate_gif,generate_jpgs,fps)
    artifacts.gif_path === nothing || println("Saved ",length(frames)+1,
        " combined frames (including the final state) to ",artifacts.gif_path)
    artifacts.jpg_directory === nothing || println("Saved ",length(artifacts.jpg_paths),
        " step JPGs to ",artifacts.jpg_directory)
    return artifacts
end

"""Compatibility wrapper returning the GIF path; also save per-step JPEGs by default."""
function render_saved_adaptive_gif(log_path::AbstractString;kwargs...)
    return render_saved_adaptive_artifacts(log_path;kwargs...).gif_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    log_path = isempty(ARGS) ? latest_adaptive_episode_log() : abspath(ARGS[1])
    filename = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(dirname(log_path),"drone_steps.gif")
    render_saved_adaptive_gif(log_path;filename)
end
