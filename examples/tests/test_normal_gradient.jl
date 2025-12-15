"""
Testing the gradient of the the normal non-gauge-preserving cost function.
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
using PEPSGauging

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
reuse_env = true

H = heisenberg_XYZ(InfiniteSquare())

gauge_tol = 1.0e-8
gauge_maxiter = 500
gauge_verbosity = 2

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

boundary_alg = SimultaneousCTMRG(;
    alg = :simultaneous,
    tol = boundary_tol,
    verbosity = boundary_verbosity,
    maxiter = boundary_maxiter,
)

gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = fpgrad_tol, maxiter = 500, verbosity = fpgrad_verbosity, krylovdim = 1000,
    ),
    iterscheme = :fixed,
)

optimizer_alg = OptimKit.LBFGS(
    32;
    gradtol = 1.0e-8,
    maxiter = 20,
    verbosity = 3,
)

## Test on random PEPS

peps0 = peps_normalize(InfinitePEPS(P, Vpeps))

# contract, gauge and contract again
env0, = leading_boundary(CTMRGEnv(peps0, Venv), peps0, boundary_alg)

# test the gradient starting from the initial PEPS
_, g0, _, _, dfs10, dfs20, = test_regular_peps_gradient(
    peps0, env0, H;
    boundary_alg,
    gradient_alg,
    symmetrization,
    steps,
)
@info "Before optimization"
@show dfs10
@show dfs20
@info "Gradient norm before optimization: $(PEPSKit.real_inner(peps0, g0, g0))"

## Test after some optimization steps

peps, env, E, = fixedpoint(
    H, peps0, env0;
    boundary_alg,
    gradient_alg,
    optimizer_alg,
    reuse_env,
    symmetrization,
)

# contract, gauge and contract again
env, = leading_boundary(CTMRGEnv(peps, Venv), peps, boundary_alg)

# test the gradient after optimization
_, g, _, _, dfs1, dfs2, = test_regular_peps_gradient(
    peps, env, H;
    boundary_alg,
    gradient_alg,
    symmetrization,
    steps,
)
@info "After optimization"
@show dfs1
@show dfs2
@info "Gradient norm after optimization: $(PEPSKit.real_inner(peps, g, g))"

## Make some plots

_, _, _, _, _, _, fig_before = test_regular_peps_gradient(
    peps0, env0, H;
    boundary_alg,
    gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

_, _, _, _, _, _, fig_after = test_regular_peps_gradient(
    peps, env, H;
    boundary_alg,
    gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

save("regular_before.png", fig_before)
save("regular_after.png", fig_after)

nothing
