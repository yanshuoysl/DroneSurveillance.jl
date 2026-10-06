using DroneSurveillance
using POMDPs

# import a solver from POMDPs.jl e.g. SARSOP
using SARSOP

# for visualization
using POMDPGifs
import Cairo

pomdp = DroneSurveillancePOMDP() # initialize the problem 

solver = SARSOPSolver(precision=1e-3) # configure the solver

policy = solve(solver, pomdp) # solve the problem

makegif(pomdp, policy, filename="out.gif")