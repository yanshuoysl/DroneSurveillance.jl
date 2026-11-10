using Test
using Random
using POMDPs
using POMDPTools
if !isdefined(@__MODULE__,:DroneSurveillanceAdaptive)
    include(joinpath(@__DIR__,"..","src","DroneSurveillanceAdaptive.jl"))
end

@testset "Configuration-stratified particle belief" begin
    DS = DroneSurveillanceAdaptive
    p = DS.DroneSurveillancePOMDP(obstacles=[DS.DSPos(3,5)])
    source = DS.initial_adaptive_state(p)
    key = DS.CountKey(p,source.config,source.mode,1,DS.DSState((1,2)),source.mode)
    entries = Pair[]
    for (r,mass) in ((DS.DSState((1,1)),0.7),(DS.DSState((2,1)),0.3)), i in 1:200
        phi = DS.DirichletCounts(Dict(key=>i))
        push!(entries,DS.adaptive_state(p,r,DS.NOMINAL_MODE,phi)=>(log(mass)-log(200)))
    end
    rare = DS.adaptive_state(p,DS.DSState((3,3)),DS.NOMINAL_MODE,DS.DirichletCounts(Dict(key=>201)))
    push!(entries,rare=>-1000.0)
    b = DS.ExactAdaptiveBelief(p,entries;logweights=true)
    before = copy(b._logprobs)
    sampled = DS.configuration_stratified_particles(b,128;rng=MersenneTwister(7))
    again = DS.configuration_stratified_particles(b,128;rng=MersenneTwister(7))
    @test sampled.expanded == 401 && sampled.retained <= 128
    @test sampled.resampled && sampled.configurations == 3
    @test sampled.belief._logprobs == again.belief._logprobs
    @test b._logprobs == before
    @test Set(k.config for k in keys(sampled.belief._logprobs)) == Set(k.config for k in keys(b._logprobs))
    function config_mass(b,r)
        foldl(DS.belief_logadd,(w for (k,w) in b._logprobs if k.config==r);init=-Inf)
    end
    for r in (DS.DSState((1,1)),DS.DSState((2,1)),DS.DSState((3,3)))
        @test config_mass(sampled.belief,r) ≈ config_mass(b,r) atol=1e-10
    end
    @test isfinite(config_mass(sampled.belief,rare.config))
    @test all(k in keys(b._logprobs) for k in keys(sampled.belief._logprobs))
    @test all(n isa Int for k in keys(sampled.belief._logprobs) for (_,n) in k.increments)
    @test actions(p,sampled.belief) == actions(p,b)
    @test sum(w for (_,w) in sampled.belief) ≈ 1
    @test !DS.configuration_stratified_particles(b,500).resampled
    @test_throws ArgumentError DS.configuration_stratified_particles(b,2)
    @test_throws ArgumentError DS.configuration_stratified_particles(b,0)

    @testset "Prepared-model filtering, budgets and atomic failure" begin
        vf = DS.RobustValueFunction(p;horizon=1,epsilon=.1,n_samples=32,seed=42)
        @test_throws ArgumentError DS.RecedingHorizonRobustPolicy(vf;H=30,particles=4)
        policy = DS.RecedingHorizonRobustPolicy(vf;H=30,particles=128,particle_seed=43)
        up = updater(policy)
        @test_throws ArgumentError action(policy,b)
        initialized = initialize_belief(up,b)
        @test length(initialized) <= 128
        @test actions(p,initialized) == actions(p,b)
        DS.reset_robust_policy!(policy)
        prior = DS.ExactAdaptiveBelief(p)
        a = action(policy,prior)
        pending = policy._policy._pending
        rng_before = rand(deepcopy(policy._particle_rng))
        @test_throws DS.ZeroObservationEvidence update(up,prior,a,DS.TERMINAL_OBSERVATION)
        @test policy.stage==0 && policy._policy._pending===pending
        @test rand(deepcopy(policy._particle_rng)) == rng_before
        @test action(policy,prior)==a
        o = DS.DSObservation(3,1,1)
        expected = DS.exact_belief_update(p,prior,a,o,pending.selected_rows)
        post = update(up,prior,a,o)
        @test post._logprobs == expected._logprobs # 14 <= budget: retain exactly
        @test policy.stage==1 && policy._pending_stage===nothing
        @test policy._filter_report.method == :configuration_stratified_particles
        @test policy._filter_report.retained==14 && !policy._filter_report.resampled
        @test all(sum(values(s.counts.increments))==1 for s in support(post))
        DS.reset_robust_policy!(policy)
        @test policy.stage==0 && policy._filter_report===nothing
        @test rand(deepcopy(policy._particle_rng)) == rng_before
        @test DS.filtering_mode(policy)==:configuration_stratified_particles
        exact = DS.RecedingHorizonRobustPolicy(vf;H=30)
        @test DS.filtering_mode(exact)==:exact_selected_model
        @test exact.particle_count===nothing
    end
end
