"""
Attempt at gauge-preserving PEPS optimization using BP gauge.

There's something wrong with the gradient, trying to figure out what...
"""

using Pkg: Pkg
Pkg.activate(pwd())

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

sd = 42039482049

## Set up test
# ------------

P = ComplexSpace(2)
Vpeps = ComplexSpace(3)
Venv = ComplexSpace(20)

symmetrization = nothing
reuse_env = true

H = heisenberg_XYZ(InfiniteSquare())
# H = transverse_field_ising(InfiniteSquare())

gauge_tol = 1.0e-8
gauge_maxiter = 500
gauge_verbosity = 2

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

optim_tol = 1.0e-5
optim_verbosity = 3
optim_maxiter = 100
optim_initial_step = 5.0e-1

gauge_alg = BeliefPropagation(;
    tol = gauge_tol,
    maxiter = gauge_maxiter,
    verbosity = gauge_verbosity,
)

gauge_gradient_alg = nothing

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

ls_alg = BackTrackingLineSearch(;
    c₁ = 1.0e-4,
    maxiter = 10,
    maxfg = 10,
    maxstep = 1.0,
)

optimizer_alg = LBFGS(
    32;
    gradtol = optim_tol,
    verbosity = optim_verbosity,
    maxiter = optim_maxiter,
    linesearch = ls_alg,
    initial_step = optim_initial_step,
)

peps0 = peps_normalize(InfinitePEPS(P, Vpeps))
peps0g, weights, bp_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0, = leading_boundary(CTMRGEnv(peps0, Venv), peps0, boundary_alg)

peps, ctmrg_env, E, = gauge_preserving_fixedpoint(
    H, peps0, ctmrg_env0, bp_env0;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg = svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    optimizer_alg,
    reuse_env,
    symmetrization,
)

jldsave("problem_peps.jld2"; peps, ctmrg_env, H)

nothing
