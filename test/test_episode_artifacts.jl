using Test
# Standalone --project=. runs also need the replay script's test-only JSON dependency.
push!(LOAD_PATH,@__DIR__)
if !isdefined(@__MODULE__, :render_saved_adaptive_artifacts)
    include(joinpath(@__DIR__,"..","examples","render_adaptive_log.jl"))
end
if !isdefined(@__MODULE__, :run_adaptive_example)
    include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))
end

@testset "Episode debugging logs and step media" begin
    mktempdir() do root
        terminal = IOBuffer()
        args = ["42","8",root,"2","--lookahead","1","--particles","128"]
        result = run_adaptive_example(args;io=terminal,warmup=false)
        console = read(result.console_log_path,String)
        @test console == String(take!(terminal))
        @test count("simulator truth BEFORE action",console) == result.steps == 8
        @test count("belief available actions:",console) == result.steps
        @test occursin("belief poses BEFORE action",console)
        @test occursin("boundary risk: 4 (move_right)",console)
        @test !occursin("boundary veto:",console)
        @test occursin("log_probability=",console)
        events = JSON.parse.(readlines(result.log_path))
        seventh = only(filter(e -> e["event"] == "decision" && e["step"] == 7,events))
        truth = only(filter(e -> e["event"] == "physical_transition" && e["step"] == 7,events))["simulator_truth"]
        @test seventh["available_actions"] == collect(1:6)
        @test truth["source"] == Dict("position"=>[2,5],"heading"=>90)
        @test truth["source_available_actions"] == collect(1:6)
        right = only(filter(b -> b["action"] == 4,seventh["boundary_risks"]))
        @test 0 < right["outward_probability"] < 1e-6
        @test all(e["available_actions"] == collect(1:6) && length(e["candidates"]) == 6
                  for e in events if e["event"] == "decision")
        @test all(haskey(c,"available_actions") for c in seventh["prior"]["configurations"])
        @test basename(result.gif_path) == "drone_steps.gif"
        @test !ispath(joinpath(result.output_directory,"drone.gif"))
        @test basename(result.jpg_directory) == "steps"
        @test length(result.jpg_paths) == result.steps + 1
        @test basename(first(result.jpg_paths)) == "step_0001.jpg"
        @test !ispath(joinpath(result.jpg_directory,"step_0000.jpg"))
        @test basename(last(result.jpg_paths)) == "step_0009.jpg"
        @test all(open(f -> read(f,3),path) == UInt8[0xff,0xd8,0xff] for path in result.jpg_paths)
        probe = JSON.parse(join(FFMPEG.exe(`-v error -count_frames -select_streams v:0
            -show_entries stream=nb_read_frames,width,height -of json $(result.gif_path)`;
            command=FFMPEG.ffprobe,collect=true),"\n"))
        @test only(probe["streams"])["nb_read_frames"] == "9"
        @test only(probe["streams"])["width"] == 1800
        @test only(probe["streams"])["height"] == 1000
        @test first(findfirst("Saved computation timing:",console)) < first(findfirst("Saved visualization",console))
        @test !ispath(joinpath(result.output_directory,"action_values.gif"))
        @test !ispath(joinpath(result.output_directory,"action_values"))
        chart_decisions = saved_action_value_decisions(events)
        @test all(r.decision.action_values == d.action_values for (r,d) in zip(result.history,chart_decisions))
        limits = action_value_chart_limits(chart_decisions)
        @test limits[1] <= minimum(c.value for d in chart_decisions for c in d.action_values)
        @test limits[2] >= maximum(c.value for d in chart_decisions for c in d.action_values)
        frames = adaptive_artifact_frames(result.history;
            initial_state=DroneSurveillanceAdaptive.initial_adaptive_state(result.pomdp))
        @test length(frames) == result.steps + 1
        @test last(frames).s == result.final_state
        for (frame,event) in zip(frames,filter(e -> e["event"] == "physical_transition",events))
            @test frame.s.mode == Tuple(Bool.(event["simulator_truth"]["source_mode"]))
        end
        @test last(frames).s.mode == Tuple(Bool.(last(filter(e ->
            e["event"] == "physical_transition",events))["simulator_truth"]["next_mode"]))
        chart_svg = IOBuffer()
        Compose.draw(Compose.SVG(chart_svg,1100 * Compose.px,720 * Compose.px),
            render_action_value_chart(chart_decisions[7],limits))
        chart_text = String(take!(chart_svg))
        @test all(occursin(s,chart_text) for s in ("Step 7","Forward","Backward","Left","Right","CW","CCW","Chosen"))
        @test occursin(string(round(seventh["candidates"][4]["value"];digits=6)),chart_text)

        # Step 7 must depict the east-facing decision pose, before its clockwise turn.
        # Compare the encoded pixels with an independently rendered known source pose.
        expected_png = joinpath(root,"expected_step7.png")
        expected_jpg = joinpath(root,"expected_step7.jpg")
        expected_pose = DroneSurveillanceAdaptive.adaptive_state(result.pomdp,
            DroneSurveillanceAdaptive.DSState((2,5),90),Tuple(Bool.(truth["source_mode"])))
        Compose.draw(Compose.PNG(expected_png,1800 * Compose.px,1000 * Compose.px),
            render_adaptive_figure(result.pomdp,
                (;t=7,s=expected_pose,phase=:before_action,decision=chart_decisions[7]),limits))
        FFMPEG.ffmpeg_exe(`-v error -nostdin -n -i $expected_png -frames:v 1
            -q:v 2 -pix_fmt yuvj420p $expected_jpg`)
        @test read(result.jpg_paths[7]) == read(expected_jpg)
        svg = IOBuffer()
        Compose.draw(Compose.SVG(svg,800 * Compose.px,800 * Compose.px),
            render(result.pomdp,(;t=7,s=expected_pose,phase=:before_action)))
        title = String(take!(svg))
        @test occursin("Step 7",title) && !occursin("before action",title)

        # Replay must reconstruct exactly the same labelled frames from saved poses.
        before = read(result.log_path)
        replay = render_saved_adaptive_artifacts(result.log_path;
            filename=joinpath(root,"replay.gif"),jpg_directory=joinpath(root,"replay_steps"))
        @test read(result.log_path) == before
        @test read(replay.gif_path) == read(result.gif_path)
        @test all(read(a) == read(b) for (a,b) in zip(replay.jpg_paths,result.jpg_paths))
        @test length(replay.jpg_paths) == length(result.jpg_paths)
        chart_only = render_saved_action_value_artifacts(result.log_path;
            filename=joinpath(root,"chart_only.gif"),jpg_directory=joinpath(root,"chart_only"))
        @test isfile(chart_only.gif_path) && length(chart_only.jpg_paths) == result.steps
        @test read(result.log_path) == before
        @test_throws ArgumentError render_saved_adaptive_artifacts(result.log_path;
            generate_gif=false,jpg_directory=result.jpg_directory)
        @test_throws ArgumentError render_saved_adaptive_artifacts(result.log_path;
            filename=result.gif_path,generate_jpgs=false)

        # Headless and JPEG-only calls still persist console diagnostics.
        short = ["42","1",root,"0","--lookahead","1"]
        headless = run_adaptive_example(short;io=nothing,generate_gif=false,warmup=false)
        @test isfile(headless.console_log_path)
        @test headless.gif_path === headless.jpg_directory === nothing
        @test isempty(headless.jpg_paths)
        stills = run_adaptive_example(short;io=nothing,generate_gif=false,generate_jpgs=true,warmup=false)
        @test stills.gif_path === nothing && length(stills.jpg_paths) == 2
        @test isempty(filter(f -> endswith(f,".gif"),readdir(stills.output_directory)))
        empty = run_adaptive_example(["42","0",root];io=nothing,warmup=false)
        @test empty.steps == 0 && empty.gif_path === nothing
        @test length(empty.jpg_paths) == 1 && isfile(only(empty.jpg_paths))
        positive = [(;t=1,action=4,action_values=[(;action=1,value=-100.0),(;action=4,value=100.0)])]
        @test action_value_chart_limits(positive) == (-100.0,100.0)
        @test_throws ArgumentError action_value_chart_limits([(;t=1,action=1,action_values=[(;action=1,value=NaN)])])
        @test_throws ArgumentError action_value_chart_limits([(;t=2,action=1,action_values=[(;action=1,value=-1.0)])])
    end
end

@testset "Final goal frame and current fault flags" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(size=(2,2),region_A=DS.DSPos(1,1),
        region_B=DS.DSPos(1,2),obstacles=DS.DSPos[])
    s = DS.initial_adaptive_state(p)
    # A fault introduced by the last transition must appear only in the final frame.
    mode = (false,true,false,false,false,false)
    goal = DS.adaptive_state(p,DS.DSState((1,2),0),mode)
    values = [(;action=a,value=Float64(a-4)) for a in 1:6]
    history = [(;t=1,s,sp=goal,a=1,decision=(;action_values=values))]
    frames = adaptive_artifact_frames(history;initial_state=s)
    @test length(frames) == 2
    @test first(frames).s == s && first(frames).s.mode == DS.NOMINAL_MODE
    @test last(frames).t == 2 && last(frames).phase == :final
    @test last(frames).s == goal && DS.is_goal(last(frames).s)
    @test last(frames).s.config.quad == p.region_B
    @test last(frames).decision == first(frames).decision
    limits = action_value_chart_limits([first(frames).decision])
    svg_text(scene) = begin
        io = IOBuffer()
        Compose.draw(Compose.SVG(io,1800 * Compose.px,1000 * Compose.px),scene)
        String(take!(io))
    end
    initial_svg = svg_text(render_adaptive_figure(p,first(frames),limits))
    final_svg = svg_text(render_adaptive_figure(p,last(frames),limits))
    @test !occursin("Fault flagged",initial_svg)
    @test !occursin("Before action",initial_svg)
    @test occursin("Final state",final_svg) && occursin("after action 1",final_svg)
    @test occursin("Goal reached",final_svg) && occursin("Last decision 1",final_svg)
    @test count("Fault flagged",final_svg) == 1
    @test count("Nominal",final_svg) == 5
    @test all(occursin(label,final_svg) for label in ("Forward","Backward","Left","Right","CW","CCW"))
    # The actual goal pose must still contain the drone's yellow nose.
    @test occursin("#FFFF00",uppercase(final_svg))
    mktempdir() do root
        artifacts = save_adaptive_artifacts(p,history;
            filename=joinpath(root,"goal.gif"),jpg_directory=joinpath(root,"steps"))
        @test length(artifacts.jpg_paths) == 2
        probe = JSON.parse(join(FFMPEG.exe(`-v error -count_frames -select_streams v:0
            -show_entries stream=nb_read_frames -of json $(artifacts.gif_path)`;
            command=FFMPEG.ffprobe,collect=true),"\n"))
        @test only(probe["streams"])["nb_read_frames"] == "2"
        expected_png = joinpath(root,"final.png")
        expected_jpg = joinpath(root,"final.jpg")
        Compose.draw(Compose.PNG(expected_png,1800 * Compose.px,1000 * Compose.px),
            render_adaptive_figure(p,last(frames),limits))
        FFMPEG.ffmpeg_exe(`-v error -nostdin -n -i $expected_png -frames:v 1
            -q:v 2 -pix_fmt yuvj420p $expected_jpg`)
        @test read(last(artifacts.jpg_paths)) == read(expected_jpg)
    end
    # Horizon/unsafe endings use their recorded state, never a fabricated goal pose.
    unsafe = DS.adaptive_state(p,DS.UNSAFE_CONFIG,mode)
    for destination in (s,unsafe)
        history = [(;t=1,s,sp=destination,a=1,decision=(;action_values=values))]
        last_frame = last(adaptive_artifact_frames(history;initial_state=s))
        @test last_frame.s == destination
        @test !occursin("Goal reached",svg_text(render_adaptive_figure(p,last_frame,limits)))
    end
end
