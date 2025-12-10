"""
Testing the gradient of the gauge-preserving cost function.
"""

using Revise

using Test
using Random
using PEPSKit
using TensorKit
using OptimKit
using KrylovKit
using Zygote
using JLD2
using BPAD

using PEPSKit: peps_normalize, BeliefPropagation, gauge_fix

include(joinpath(pwd(), "tools.jl"))

sd = 42039482049

## Set up test
# ------------

steps = -1.0e-5:(2 * 1.0e-5):1.0e-5 # zoom
steps_plot = LinRange(-0.05, 0.5, 21) # overview

P = ComplexSpace(2)
Vpeps = ComplexSpace(3)
Venv = ComplexSpace(20)

symmetrization = nothing

H = heisenberg_XYZ(InfiniteSquare())

gauge_tol = 1.0e-8
gauge_maxiter = 500
gauge_verbosity = 2

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

gauge_alg = BeliefPropagation(;
    tol = gauge_tol,
    maxiter = gauge_maxiter,
    verbosity = gauge_verbosity,
)

svd_alg = SVDAdjoint(;
    fwd_alg = (; alg = :sdd),
    rrule_alg = (; alg = :full),
)

boundary_alg = SimultaneousCTMRG(;
    alg = :simultaneous,
    tol = boundary_tol,
    verbosity = boundary_verbosity,
    maxiter = boundary_maxiter,
)

boundary_gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = fpgrad_tol, maxiter = 500, verbosity = fpgrad_verbosity, krylovdim = 1000,
    ),
    iterscheme = :fixed,
)

## Test on random PEPS

peps0 = peps_normalize(InfinitePEPS(P, Vpeps))

# contract, gauge and contract again
ctmrg_env0, = leading_boundary(CTMRGEnv(peps0, Venv), peps0, boundary_alg)
peps0g, weights, bp_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0g, = leading_boundary(CTMRGEnv(peps0g, Venv), peps0g, boundary_alg)

# test effect of gauging: network value and energy expectation value
n0 = network_value(peps0, ctmrg_env0)
n0g = network_value(peps0g, ctmrg_env0g)
e0 = expectation_value(peps0, H, ctmrg_env0)
e0g = expectation_value(peps0g, H, ctmrg_env0g)

@info "n0 = \t$n0"
@info "n0g = \t$n0g"
@info "e0 = \t$e0"
@info "e0g = \t$e0g"

# test the gradient starting from the ungauged PEPS
_, g0_before_nogauge, _, _, dfs1_before_nogauge, dfs2_before_nogauge, = test_peps_gradient(
    peps0, ctmrg_env0, bp_env0, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps,
)
@info "Before optimization, ungauged PEPS"
@show dfs1_before_nogauge
@show dfs2_before_nogauge
@info "Gradient norm before optimization, ungauged PEPS: $(PEPSKit.real_inner(peps0, g0_before_nogauge, g0_before_nogauge))"

# test the gradient starting from the gauged PEPS
_, g0_before_gauge, _, _, dfs1_before_gauge, dfs2_before_gauge, = test_peps_gradient(
    peps0g, ctmrg_env0g, bp_env0, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps,
)
@info "Before optimization, gauged PEPS"
@show dfs1_before_gauge
@show dfs2_before_gauge
@info "Gradient norm before optimization, gauged PEPS: $(PEPSKit.real_inner(peps0g, g0_before_gauge, g0_before_gauge))"


## Test after some optimization steps

data = load("problem_peps.jld2")
peps = data["peps"]
ctmrg_env = data["ctmrg_env"]
H = data["H"]

# contract, gauge and contract again
ctmrg_env, = leading_boundary(CTMRGEnv(peps, Venv), peps, boundary_alg)
pepsg, weights, bp_env = gauge_fix(peps, gauge_alg)
ctmrg_envg, = leading_boundary(CTMRGEnv(pepsg, Venv), pepsg, boundary_alg)

# test effect of gauging: network value and energy expectation value
n = network_value(peps, ctmrg_env)
ng = network_value(pepsg, ctmrg_envg)
e = expectation_value(peps, H, ctmrg_env)
eg = expectation_value(pepsg, H, ctmrg_envg)

@info "n = \t$n"
@info "ng = \t$ng"
@info "e = \t$e"
@info "eg = \t$eg"

# test the gradient starting from the ungauged PEPS
_, g0_after_nogauge, _, _, dfs1_after_nogauge, dfs2_after_nogauge, = test_peps_gradient(
    peps, ctmrg_env, bp_env, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps,
)
@info "After optimization, ungauged PEPS"
@show dfs1_after_nogauge
@show dfs2_after_nogauge
@info "Gradient norm after optimization, ungauged PEPS: $(PEPSKit.real_inner(peps, g0_after_nogauge, g0_after_nogauge))"

# test the gradient starting from the gauged PEPS
_, g0_after_gauge, _, _, dfs1_after_gauge, dfs2_after_gauge, = test_peps_gradient(
    pepsg, ctmrg_envg, bp_env, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps,
)
@info "After optimization, gauged PEPS"
@show dfs1_after_gauge
@show dfs2_after_gauge
@info "Gradient norm after optimization, gauged PEPS: $(PEPSKit.real_inner(pepsg, g0_after_gauge, g0_after_gauge))"


## Make some plots

# test the gradient starting from the ungauged PEPS
_, _, _, _, _, _, fig_before_nogauge = test_peps_gradient(
    peps0, ctmrg_env0, bp_env0, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)


# test the gradient starting from the gauged PEPS
_, g0_before_gauge, _, _, _, _, fig_before_gauge = test_peps_gradient(
    peps0g, ctmrg_env0g, bp_env0, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)
@info "Gradient norm before optimization, ungauged PEPS: $(PEPSKit.real_inner(peps0, g0_before_nogauge, g0_before_nogauge))"

# test the gradient starting from the ungauged PEPS
_, _, _, _, _, _, fig_after_nogauge = test_peps_gradient(
    peps, ctmrg_env, bp_env, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

# test the gradient starting from the gauged PEPS
_, _, _, _, _, _, fig_after_gauge = test_peps_gradient(
    pepsg, ctmrg_envg, bp_env, H;
    gauge_alg,
    gauge_gradient_alg = nothing,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

save("bp_before_nogauge.png", fig_before_nogauge)
save("bp_before_gauge.png", fig_before_gauge)
save("bp_after_nogauge.png", fig_after_nogauge)
save("bp_after_gauge.png", fig_after_gauge)

nothing
