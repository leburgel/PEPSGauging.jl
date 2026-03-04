"""
Benchmark BP-gauge-fixed optimization on the square lattice Heisenberg model.
"""

using Revise

using Random
using PEPSKit
using TensorKit
using OptimKit
using KrylovKit
using JLD2
using LinearAlgebra
using PEPSGauging

include(joinpath(pwd(), "tools.jl"))

using PEPSKit: peps_normalize, BeliefPropagation, gauge_fix

sd = 12345678 # trial 3
BLAS.set_num_threads(4) # be a bit conservative

# SETUP
# -----

T = ComplexF64
D = 3
chi = 20
chi1 = 50
symmetrization = nothing # RotateReflect()
reuse_env = true
Jx = -1.0
Jy = 1.0
Jz = -1.0

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

optim_tol = 1.0e-8
optim_verbosity = 3
optim_maxiter = 700

H = heisenberg_XYZ(InfiniteSquare(); Jx, Jy, Jz)

P = H.lattice
Vpeps = fill(ComplexSpace(D), size(P)...)
Venv = fill(ComplexSpace(chi), size(P)...)
Venv1 = fill(ComplexSpace(chi1), size(P)...)

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

optimizer_alg = LBFGS(
    32; gradtol = optim_tol, verbosity = optim_verbosity, maxiter = optim_maxiter
)


# MAIN
# ----

Random.seed!(sd)

es0 = T[]
es1 = T[]
ngs = real(T)[]

peps0 = peps_normalize(symmetrize!(InfinitePEPS(randn, T, P, Vpeps), symmetrization))
env0, = leading_boundary(CTMRGEnv(randn, T, peps0, Venv), peps0, boundary_alg)
env1 = CTMRGEnv(randn, T, peps0, Venv1)

fname = joinpath(
    pwd(),
    "results",
    generate_heisenberg_filename(D, chi, chi1, nothing, symmetrization, Jx, Jy, Jz),
)

nogauge_finalize! = nogauge_energy_tracker(
    es0, es1, ngs, H, env1;
    boundary_alg, save_iter = 10, fname,
)

peps, env, E, = fixedpoint(
    H, peps0, env0;
    boundary_alg,
    gradient_alg,
    optimizer_alg,
    reuse_env,
    symmetrization,
    (finalize!) = (nogauge_finalize!),
)

jldsave(
    fname;
    peps,
    env,
    es0,
    es1,
    ngs,
)

nothing
