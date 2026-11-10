# Review src/DroneSurveillanceAdaptive.jl before the first integrated run.
# julia --project=. examples/run_adaptive.jl [seed=42] [H=2] [output_root] [N_f=2] [--lookahead L=2] [--particles N] [--fault_time_step T] [--fault_type A] [warmup=false] [tied_action_rdn=false]
# Each run gets a new folder inside output_root (default: <repository>/test/output).
# Saves logs, rewards, and combined map/action-value/fault figures in drone_steps.gif/steps/*.jpg.
# Optional warmup=true replays compilation work; normal runs skip the replay.
using POMDPs
using Random
if !isdefined(@__MODULE__, :DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__, "..", "src", "DroneSurveillanceAdaptive.jl"))
end

if !isdefined(@__MODULE__, :AdaptiveEpisodeConsole)
    """Stream an episode's console output to disk and optionally to the caller."""
    struct AdaptiveEpisodeConsole <: IO
        file::IO
        mirror::Union{Nothing,IO}
    end
end
function Base.unsafe_write(io::AdaptiveEpisodeConsole, p::Ptr{UInt8}, n::UInt)
    written = Base.unsafe_write(io.file,p,n)
    io.mirror === nothing || Base.unsafe_write(io.mirror,p,n)
    return written
end
function Base.write(io::AdaptiveEpisodeConsole, byte::UInt8)
    written = write(io.file,byte)
    io.mirror === nothing || write(io.mirror,byte)
    return written
end
function Base.flush(io::AdaptiveEpisodeConsole)
    flush(io.file)
    io.mirror === nothing || flush(io.mirror)
    return nothing
end

"""Parse positional arguments, planning/fault settings, and boolean run options."""
function adaptive_example_settings(args)
    positional = String[]
    lookahead, specified, particles, particle_specified, i = 2, false, nothing, false, 1
    fault_time_step = nothing
    fault_type = nothing
    booleans = Dict("warmup"=>false,"tied_action_rdn"=>false)
    boolean_specified = Set{String}()
    while i <= length(args)
        arg = String(args[i])
        # Accept name=value and --name[=value] without consuming a positional slot.
        option_name = replace(first(split(arg,'=';limit=2)),r"^--"=>"")
        if haskey(booleans,option_name) && (occursin('=',arg) || startswith(arg,"--"))
            option_name in boolean_specified && throw(ArgumentError("specify $option_name only once"))
            push!(boolean_specified,option_name)
            value = if !occursin('=',arg)
                i < length(args) || throw(ArgumentError("--$option_name requires true or false"))
                i += 1
                String(args[i])
            else
                split(arg,'=';limit=2)[2]
            end
            lowercase(value) in ("true","false") ||
                throw(ArgumentError("$option_name must be true or false, got $(repr(value))"))
            booleans[option_name] = lowercase(value) == "true"
        elseif arg == "--lookahead" || startswith(arg,"--lookahead=")
            specified && throw(ArgumentError("specify --lookahead only once"))
            specified = true
            if arg == "--lookahead"
                i < length(args) || throw(ArgumentError("--lookahead requires an integer"))
                i += 1
                lookahead = parse(Int,args[i])
            else
                lookahead = parse(Int,split(arg,'=';limit=2)[2])
            end
        elseif arg == "--particles" || startswith(arg,"--particles=")
            particle_specified && throw(ArgumentError("specify --particles only once"))
            particle_specified = true
            if arg == "--particles"
                i < length(args) || throw(ArgumentError("--particles requires an integer"))
                i += 1
                particles = parse(Int,args[i])
            else
                particles = parse(Int,split(arg,'=';limit=2)[2])
            end
        elseif arg == "--fault_time_step" || startswith(arg,"--fault_time_step=")
            fault_time_step === nothing || throw(ArgumentError("specify --fault_time_step only once"))
            if arg == "--fault_time_step"
                i < length(args) || throw(ArgumentError("--fault_time_step requires an integer"))
                i += 1
                fault_time_step = parse(Int,args[i])
            else
                fault_time_step = parse(Int,split(arg,'=';limit=2)[2])
            end
        elseif arg == "--fault_type" || startswith(arg,"--fault_type=")
            fault_type === nothing || throw(ArgumentError("specify --fault_type only once"))
            if arg == "--fault_type"
                i < length(args) || throw(ArgumentError("--fault_type requires an integer action ID"))
                i += 1
                fault_type = parse(Int,args[i])
            else
                fault_type = parse(Int,split(arg,'=';limit=2)[2])
            end
        elseif startswith(arg,"--")
            throw(ArgumentError("unknown option: $arg"))
        else
            push!(positional,arg)
        end
        i += 1
    end
    length(positional) <= 4 || throw(ArgumentError("expected seed, H, output_root, N_f and optional --lookahead L, --particles N, --fault_time_step T, --fault_type A, warmup=true|false, tied_action_rdn=true|false"))
    seed = length(positional) >= 1 ? parse(Int,positional[1]) : 42
    H = length(positional) >= 2 ? parse(Int,positional[2]) : 2
    output_root = length(positional) >= 3 ? abspath(positional[3]) :
        normpath(joinpath(@__DIR__,"..","test","output"))
    N_f = length(positional) >= 4 ? parse(Int,positional[4]) : 2
    seed >= 0 && H >= 0 && N_f >= 0 || throw(ArgumentError("seed, H and N_f must be nonnegative"))
    lookahead > 0 || throw(ArgumentError("planning lookahead L must be positive"))
    particles === nothing || particles > 0 || throw(ArgumentError("particle count must be positive"))
    if fault_time_step !== nothing
        1 <= fault_time_step <= H || throw(ArgumentError("fault_time_step must be inside 1:H"))
        N_f >= 1 || throw(ArgumentError("fault_time_step requires N_f >= 1"))
    end
    if fault_type !== nothing
        fault_type in 1:6 || throw(ArgumentError("fault_type must be an action ID from 1 to 6"))
        fault_time_step === nothing && throw(ArgumentError("--fault_type requires --fault_time_step"))
    end
    return (;seed,H,output_root,N_f,lookahead,particles,fault_time_step,fault_type,
        warmup=booleans["warmup"],tied_action_rdn=booleans["tied_action_rdn"])
end

function adaptive_example_problem(settings)
    DS = DroneSurveillanceAdaptive
    pomdp = DS.DroneSurveillancePOMDP(region_A=DS.DSPos(1,1),obstacles=[DS.DSPos(3,5)])
    belief = DS.ExactAdaptiveBelief(pomdp)
    vf = DS.RobustValueFunction(pomdp;horizon=settings.lookahead,epsilon=0.1,
        n_samples=512,seed=settings.seed)
    policy = DS.RecedingHorizonRobustPolicy(vf;H=settings.H,particles=settings.particles,
        particle_seed=settings.seed+1,tied_action_rdn=settings.tied_action_rdn)
    return (;pomdp,belief,policy)
end

function warm_up_adaptive_example(settings)
    (;pomdp,belief,policy) = adaptive_example_problem(settings)
    # Match the measured runner's logger type, but write no files or terminal text.
    result = DroneSurveillanceAdaptive.simulate_robust(pomdp,policy;
        H=settings.H,N_f=settings.N_f,seed=settings.seed,initial_belief=belief,io=devnull,
        fault_steps=settings.fault_time_step === nothing ? nothing : [settings.fault_time_step],
        fault_actions=settings.fault_type === nothing ? nothing : [settings.fault_type])
    return (;steps=result.steps,reason=result.reason)
end

"""AR episode with independent budget H and receding planning lookahead L.
Each decision fits a fresh local problem from the current weighted belief.
Default filtering is exact; --particles N explicitly approximates mode/count
histories while preserving configuration support. Each retained phi is a full
integer-count history. Local horizon min(L,remaining steps) has zero terminal value.
No route/waypoint, true-state policy input, or posterior-mean fallback is used.
Warm-up is disabled by default. On the command line, use warmup=true or warmup=false
(also --warmup=true / --warmup false). For benchmarking, use warmup=true to replay
the same seeded episode before constructing a fresh measured episode. No learned
counts, caches, or RNG state carry across. Residual compilation is reported with
or without warm-up. timing.json separates core computation from I/O.
An explicit programmatic warmup keyword overrides the command-line setting.
tied_action_rdn=false keeps the first numerical maximum in action order.
Use tied_action_rdn=true (also --tied_action_rdn=true / --tied_action_rdn true)
to sample uniformly among numerical maxima with a separate seeded RNG.
With --fault_time_step T, schedule exactly one fault at step T, choosing a
random healthy action before executing that command. T must be inside 1:H and
N_f must be positive. Omit the option to retain the random fault schedule.
Add --fault_type A to select the permanently disabled action: 1 Forward,
2 Backward, 3 Left, 4 Right, 5 CW, 6 CCW. This requires --fault_time_step;
the fault is activated before that step's command and remains active afterward.
console.log is always saved, including when io=nothing suppresses terminal output.
By default save combined figures in drone_steps.gif and steps/step_NNNN.jpg:
drone map on the left, recorded Q-values on the right, current fault flags below.
Decision frames show the pose BEFORE the numbered action. Frame N+1 shows the
final destination and fault mode, retaining the last decision's labelled values.
An empty episode saves only step_0000.jpg (initial state) and no GIF.
Pass generate_gif=false for a headless run; generate_jpgs defaults to the same
setting but can be independently enabled. Rendering starts after timing is saved.
"""
function run_adaptive_example(args=ARGS; io=stdout, generate_gif::Bool=true,
        generate_jpgs::Bool=generate_gif, warmup::Union{Nothing,Bool}=nothing,
        tied_action_rdn::Union{Nothing,Bool}=nothing)
    settings = adaptive_example_settings(args)
    tied_action_rdn = tied_action_rdn === nothing ? settings.tied_action_rdn : tied_action_rdn
    settings = (;settings...,tied_action_rdn)
    warmup = warmup === nothing ? settings.warmup : warmup
    (;seed,H,output_root,N_f,lookahead,particles,fault_time_step,fault_type) = settings
    # Check rendering dependencies before fitting or executing an episode.
    # Import lazily so merely including this script does not load the backend.
    if (generate_gif || generate_jpgs) && !isdefined(@__MODULE__, :save_adaptive_artifacts)
        include(joinpath(@__DIR__,"adaptive_artifacts.jl"))
    end
    DS = DroneSurveillanceAdaptive
    mkpath(output_root)
    # Atomically reserve a unique folder, even for repeated/concurrent seeds.
    # Keep it after Julia exits so every episode's records remain available.
    directory = mktempdir(output_root;prefix="run_seed$(seed)_H$(H)_L$(lookahead)_$(particles === nothing ? "exact" : "particles$(particles)")_",cleanup=false)
    console_log_path = joinpath(directory,"console.log")
    return open(console_log_path,"w") do console_file
        console = AdaptiveEpisodeConsole(console_file,io)
        warmup_report = if warmup
            println(console,"Compilation warm-up: replaying the same seeded episode; excluded from measured computation time.")
            report = warm_up_adaptive_example(settings)
            GC.gc() # Reclaim the warm-up history outside the measured workload.
            report
        else
            nothing
        end
        # Fresh policy, counts, HDR caches, and RNGs: warm-up cannot train this episode.
        (;pomdp,belief,policy) = adaptive_example_problem(settings)
        paths = [joinpath(directory,name) for name in ("episode.jsonl","rewards.csv","rewards.svg")]
        println(console,"Receding-horizon planning: episode H=",H,
            "; lookahead L=",lookahead,"; particles=",particles,"; seed=",seed,"; N_f=",N_f,
            "; tied_action_rdn=",tied_action_rdn,"; fault_time_step=",fault_time_step,
            "; fault_type=",fault_type,
            "; output=",directory)
        result = DS.simulate_robust(pomdp,policy;H,N_f,seed,initial_belief=belief,io=console,
            fault_steps=fault_time_step === nothing ? nothing : [fault_time_step],
            fault_actions=fault_type === nothing ? nothing : [fault_type],
            log_path=paths[1],reward_csv_path=paths[2],reward_chart_path=paths[3],
            timing_warmup_completed=warmup)
        timing_path = joinpath(directory,"timing.json")
        timing = (;result.timing...,warmup_completed=warmup,warmup=warmup_report,
            julia_version=string(VERSION))
        open(timing_path,"w") do file
            DS.episode_json(file,timing)
            println(file)
        end
        println(console,"Saved log: ",result.log_path,"\nSaved rewards: ",result.reward_csv_path,
            "\nSaved reward chart: ",result.reward_chart_path)
        println(console,"Saved computation timing: ",timing_path)
        artifacts = if generate_gif || generate_jpgs
            Base.invokelatest(() -> save_adaptive_artifacts(pomdp,result.history;
                filename=joinpath(directory,"drone_steps.gif"),
                jpg_directory=joinpath(directory,"steps"),generate_gif,generate_jpgs))
        else
            (;gif_path=nothing,jpg_directory=nothing,jpg_paths=String[])
        end
        artifacts.gif_path === nothing || println(console,"Saved visualization to ",artifacts.gif_path)
        artifacts.jpg_directory === nothing || println(console,"Saved ",length(artifacts.jpg_paths),
            " step JPGs to ",artifacts.jpg_directory)
        generate_gif && isempty(result.history) && println(console,"No actions executed; GIF skipped.")
        println(console,"Saved console log: ",console_log_path)
        flush(console)
        return (;result...,timing,timing_path,console_log_path,output_directory=directory,
            artifacts...,pomdp,policy,episode_horizon=H,planning_lookahead=lookahead,
            particle_count=particles,tied_action_rdn,fault_time_step,fault_type,fit_report=policy._fit_report)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_adaptive_example()
end
