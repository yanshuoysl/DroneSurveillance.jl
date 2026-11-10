# Shared by live episodes and saved-log replay. Rendering is outside core timing.
import Compose, Cairo, Fontconfig, FFMPEG
using POMDPTools: render

"""Build decision frames plus the actual destination after the last action."""
function adaptive_artifact_frames(history; initial_state)
    records = collect(history)
    isempty(records) && return [(;t=0,s=initial_state,phase=:initial,decision=nothing)]
    [Int(r.t) for r in records] == collect(1:length(records)) ||
        throw(ArgumentError("action steps must be consecutive starting at one"))
    frames = [(;t=Int(r.t),s=r.s,phase=:before_action,
        decision=get(r,:decision,nothing) === nothing ? nothing :
            (;t=Int(r.t),action=Int(r.a),action_values=r.decision.action_values)) for r in records]
    push!(frames,(;t=length(records)+1,s=last(records).sp,phase=:final,
        decision=last(frames).decision))
    return frames
end

"""Save combined map/Q-value/fault figures as one GIF and one JPEG per frame.
Step t precedes action t; frame N+1 shows the destination after action N with
the last decision's values explicitly labelled. Empty histories save only an
initial JPEG. Existing outputs are rejected; temporary PNGs are cleaned up.
"""
function save_adaptive_artifacts(pomdp, history;
        filename::AbstractString, jpg_directory::AbstractString,
        initial_state=DroneSurveillanceAdaptive.initial_adaptive_state(pomdp),
        generate_gif::Bool=true, generate_jpgs::Bool=true, fps::Integer=2)
    frames = adaptive_artifact_frames(history;initial_state)
    decisions = [f.decision for f in frames if f.phase == :before_action]
    limits = action_value_chart_limits(decisions)
    return save_adaptive_frame_sequence(frames,step -> render_adaptive_figure(pomdp,step,limits);
        filename,jpg_directory,generate_gif=generate_gif && first(frames).phase != :initial,
        generate_jpgs,fps,width=1800,height=1000)
end

"""Encode identical step-numbered PNG sources into GIF and JPEG outputs."""
function save_adaptive_frame_sequence(frames,render_frame;
        filename::AbstractString,jpg_directory::AbstractString,
        generate_gif::Bool=true,generate_jpgs::Bool=true,fps::Integer=2,
        width::Integer=800,height::Integer=800)
    fps > 0 || throw(ArgumentError("fps must be positive"))
    write_gif = generate_gif && !isempty(frames)
    write_jpgs = generate_jpgs && !isempty(frames)
    write_gif && ispath(filename) && throw(ArgumentError("GIF already exists: $filename"))
    write_jpgs && ispath(jpg_directory) &&
        throw(ArgumentError("step image directory already exists: $jpg_directory"))
    !write_gif && !write_jpgs && return (;gif_path=nothing,jpg_directory=nothing,jpg_paths=String[])
    frame_steps = [Int(frame.t) for frame in frames]
    first_step = first(frame_steps)
    frame_steps == collect(first_step:first_step+length(frames)-1) ||
        throw(ArgumentError("rendered steps must be consecutive"))
    mktempdir() do temporary
        for step in frames
            png = joinpath(temporary,"step_$(lpad(step.t,4,'0')).png")
            Compose.draw(Compose.PNG(png,width * Compose.px,height * Compose.px),render_frame(step))
        end
        pattern = joinpath(temporary,"step_%04d.png")
        count = length(frames)
        if write_gif
            mkpath(dirname(abspath(filename)))
            palette = "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse"
            FFMPEG.ffmpeg_exe(`-v error -nostdin -n -framerate $fps -start_number $first_step
                -i $pattern -filter_complex $palette -frames:v $count -loop 0 $filename`)
        end
        if write_jpgs
            mkpath(jpg_directory)
            output = joinpath(jpg_directory,"step_%04d.jpg")
            FFMPEG.ffmpeg_exe(`-v error -nostdin -n -start_number $first_step -i $pattern
                -frames:v $count -q:v 2 -pix_fmt yuvj420p -start_number $first_step $output`)
        end
    end
    jpg_paths = write_jpgs ? [abspath(joinpath(jpg_directory,"step_$(lpad(t,4,'0')).jpg"))
                                for t in frame_steps] : String[]
    return (;gif_path=write_gif ? abspath(filename) : nothing,
        jpg_directory=write_jpgs ? abspath(jpg_directory) : nothing,jpg_paths)
end

"""Validate decision data and derive one shared Q-value scale for the episode."""
function action_value_chart_limits(decisions)
    values = Float64[]
    for (t,decision) in enumerate(decisions)
        decision === nothing && continue
        decision.t == t || throw(ArgumentError("decision steps must be consecutive starting at one"))
        ids = [Int(c.action) for c in decision.action_values]
        !isempty(ids) && all(a -> 1 <= a <= 6,ids) && length(unique(ids)) == length(ids) ||
            throw(ArgumentError("each decision needs distinct action IDs from 1:6"))
        decision.action in ids || throw(ArgumentError("selected action has no logged value"))
        all(c -> isfinite(c.value),decision.action_values) ||
            throw(ArgumentError("action values must be finite"))
        append!(values,Float64[c.value for c in decision.action_values])
    end
    isempty(values) && return (-1.0,1.0)
    low,high = min(0.0,minimum(values)),max(0.0,maximum(values))
    low == high && return (-1.0,1.0)
    # Round to readable tick intervals while retaining a zero baseline.
    rough = (high-low)/5
    magnitude = 10.0^floor(log10(rough))
    step = first(m for m in (1.0,2.0,5.0,10.0) if m*magnitude >= rough)*magnitude
    return (floor(low/step)*step,ceil(high/step)*step)
end

"""Actions along the x-axis, with Q-value bars extending up/down from zero."""
function render_action_value_chart(decision,limits; title="Step $(decision.t)")
    C = Compose
    low,high = limits
    x(a) = 0.19 + (a-1)*0.14
    y(v) = 0.75 - 0.54*(v-low)/(high-low)
    labels = ("1  Forward","2  Backward","3  Left","4  Right","5  CW","6  CCW")
    selected = "#15803d"
    other = "#2563eb"
    muted = "#64748b"
    parts = Any[]
    label(px,py,value;size=13,color="#0f172a",align=C.hleft) =
        C.compose(C.context(),C.text(px,py,value,align,C.vcenter),
            C.font("DejaVu Sans"),C.fontsize(size * C.pt),C.fill(color))
    push!(parts,label(0.5,0.055,title;size=20,align=C.hcenter))
    push!(parts,label(0.5,0.115,"Robust Q-value by action";
        size=13,color=muted,align=C.hcenter))
    push!(parts,label(0.025,0.175,"Q-value";size=12,color=muted))
    # Construct foreground objects before backgrounds; Compose paints first on top.
    by_action = Dict(Int(c.action)=>Float64(c.value) for c in decision.action_values)
    for a in 1:6
        color = a == decision.action ? selected : other
        push!(parts,label(x(a),0.835,labels[a];size=13,
            color=a == decision.action ? selected : "#0f172a",align=C.hcenter))
        if haskey(by_action,a)
            value = by_action[a]
            label_y = y(value) + (value < 0 ? 0.027 : -0.027)
            push!(parts,label(x(a),label_y,string(round(value;digits=6));
                size=12,color=color,align=C.hcenter))
            push!(parts,C.compose(C.context(),C.rectangle(x(a)-0.0375,min(y(0),y(value)),
                0.075,abs(y(value)-y(0))),C.fill(color),C.stroke(nothing)))
        else
            push!(parts,label(x(a),y(0)+0.04,"Not evaluated";size=10,color=muted,align=C.hcenter))
        end
    end
    push!(parts,C.compose(C.context(),C.line([(0.12,y(0)),(0.96,y(0))]),
        C.stroke("#475569"),C.linewidth(0.45 * C.mm)))
    for value in range(low,high;length=6)
        push!(parts,label(0.095,y(value),string(round(value;digits=2));size=10,color=muted,align=C.hright))
        push!(parts,C.compose(C.context(),C.line([(0.12,y(value)),(0.96,y(value))]),
            C.stroke("#e2e8f0"),C.linewidth(0.2 * C.mm)))
    end
    push!(parts,label(0.54,0.9,"Action";size=12,color=muted,align=C.hcenter))
    push!(parts,label(0.5,0.965,"Chosen action: $(labels[decision.action])";
        size=13,color=selected,align=C.hcenter))
    push!(parts,C.compose(C.context(),C.rectangle(),C.fill("white"),C.stroke(nothing)))
    return C.compose(C.context(),parts...)
end

"""Six fault flags for the displayed physical state, in action order."""
function render_fault_mode(mode)
    C = Compose
    labels = ("Forward","Backward","Left","Right","CW","CCW")
    parts = Any[]
    push!(parts,C.compose(C.context(),
        C.text(0.5,0.14,"Current fault mode",C.hcenter,C.vcenter),
        C.font("DejaVu Sans"),C.fontsize(14 * C.pt),C.fill("#334155")))
    for a in 1:6
        x = (a-1)/6 + 0.008
        flagged = mode[a]
        color = flagged ? "#b91c1c" : "#15803d"
        status = flagged ? "Fault flagged" : "Nominal"
        push!(parts,C.compose(C.context(),
            C.text(x+0.075,0.5,labels[a],C.hcenter,C.vcenter),
            C.text(x+0.075,0.77,status,C.hcenter,C.vcenter),
            C.font("DejaVu Sans"),C.fontsize(13 * C.pt),C.fill(color)))
        push!(parts,C.compose(C.context(),C.rectangle(x,0.33,0.15,0.62),
            C.fill(flagged ? "#fef2f2" : "#f0fdf4"),C.stroke(color),
            C.linewidth(0.3 * C.mm)))
    end
    return C.compose(C.context(),parts...)
end

"""Map on the left, recorded decision values on the right, current faults below."""
function render_adaptive_figure(pomdp,frame,limits)
    C = Compose
    DS = DroneSurveillanceAdaptive
    final = frame.phase == :final
    title = frame.phase == :initial ? "Initial state (no actions)" :
        final ? "Final state — after action $(frame.t-1)" : "Step $(frame.t)"
    status = DS.is_goal(frame.s) ? "Goal reached" :
        DS.is_unsafe(frame.s) ? "Unsafe state" : frame.phase == :initial ? "No actions executed" :
        final ? "Episode complete" : nothing
    text_label(x,y,value,size,color) = C.compose(C.context(),
        C.text(x,y,value,C.hcenter,C.vcenter),C.font("DejaVu Sans"),
        C.fontsize(size * C.pt),C.fill(color))
    chart = frame.decision === nothing ?
        text_label(0.5,0.5,"No action values recorded",17,"#64748b") :
        render_action_value_chart(frame.decision,limits;
            title=final ? "Last decision $(frame.decision.t) — action values" :
                "Decision $(frame.decision.t) — action values")
    return C.compose(C.context(),
        text_label(0.5,0.04,title,24,"#0f172a"),
        status === nothing ? nothing : text_label(0.5,0.085,status,14,"#64748b"),
        C.compose(C.context(0.015,0.12,0.435,0.71),render(pomdp,(;s=frame.s))),
        C.compose(C.context(0.46,0.12,0.53,0.71),chart),
        C.compose(C.context(0.02,0.85,0.96,0.14),render_fault_mode(frame.s.mode)),
        C.compose(C.context(),C.rectangle(),C.fill("white"),C.stroke(nothing)))
end

"""Export one Q-value bar chart per decision, aligned with the drone's step numbers.
Values are supplied from recorded decisions, never recomputed from rewards.
Empty episodes have no decision charts. Missing legacy candidates are labelled
Not evaluated rather than assigned fabricated zero-valued bars.
"""
function save_action_value_artifacts(decisions;
        filename::AbstractString,jpg_directory::AbstractString,
        generate_gif::Bool=true,generate_jpgs::Bool=true,fps::Integer=2)
    frames = collect(decisions)
    limits = action_value_chart_limits(frames)
    return save_adaptive_frame_sequence(frames,d -> render_action_value_chart(d,limits);
        filename,jpg_directory,generate_gif,generate_jpgs,fps,width=1100,height=720)
end
