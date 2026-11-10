# After reviewing src/DroneSurveillanceAdaptive.jl:
# julia --project=test test/test_visualization_adaptive.jl [seed=42] [H=2] [output_root] [N_f=2] [--lookahead L=2]
# Compatibility entry point; the main example now generates the GIF itself.
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..")))
include(joinpath(@__DIR__,"..","examples","run_adaptive.jl"))

function visualize_adaptive_example(args=ARGS)
    return run_adaptive_example(args;generate_gif=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    visualize_adaptive_example()
end
