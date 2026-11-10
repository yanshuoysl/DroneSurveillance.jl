julia --project=. ./examples/run_adaptive.jl 42 30 --lookahead 1 --particles 128 warmup=false

julia --project=. examples/run_adaptive.jl 38 30 --lookahead 2 --particles 128 tied_action_rdn=false