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
using PEPSGauging: MCF, mcf_environment

sd = 1234 # trial 1
# sd = 123456 # trial 2
# sd = 12345678 # trial 3
BLAS.set_num_threads(4) # be a bit conservative

# SETUP
# -----

lattice = InfiniteSquare(2, 2)

T = ComplexF64
D = 3
chi = 20
chi1 = 50
symmetrization = nothing
reuse_env = true
Jx = -1.0
Jy = 1.0
Jz = -1.0

gauge_tol = 1.0e-8
gauge_maxiter = 500
gauge_verbosity = 2

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

optim_tol = 1.0e-7
optim_verbosity = 3
optim_maxiter = 1200

H = heisenberg_XYZ(lattice; Jx, Jy, Jz)

Vpeps = fill(ComplexSpace(D), size(lattice)...)
Venv = fill(ComplexSpace(chi), size(lattice)...)
Venv1 = fill(ComplexSpace(chi1), size(lattice)...)

gauge_alg = MCF(;
    tol = gauge_tol,
    maxiter = gauge_maxiter,
    verbosity = gauge_verbosity,
)

gauge_gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = fpgrad_tol, maxiter = 500, verbosity = fpgrad_verbosity, krylovdim = 1000,
    ),
    iterscheme = :characteristic,
)

svd_alg = nothing # DUMMY

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

optimizer_alg = LBFGS(
    32; gradtol = optim_tol, verbosity = optim_verbosity, maxiter = optim_maxiter
)


# MCF
# ---

Random.seed!(sd)

es0 = T[]
es1 = T[]
ngs = real(T)[]

peps0 = peps_normalize(symmetrize!(InfinitePEPS(randn, T, physicalspace(H), Vpeps), symmetrization))
peps0, weights, mcf_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0, = leading_boundary(CTMRGEnv(randn, T, peps0, Venv), peps0, boundary_alg)
ctmrg_env1 = CTMRGEnv(randn, T, peps0, Venv1)

fname = joinpath(
    pwd(),
    "results",
    generate_heisenberg_filename(D, chi, chi1, "mcf_fixed", symmetrization, Jx, Jy, Jz),
)

gauge_finalize! = gauge_energy_tracker(
    es0, es1, ngs, H, ctmrg_env1;
    boundary_alg, gauge_alg, save_iter = 10, fname,
)

peps, env, E, = gauge_preserving_fixedpoint(
    H, peps0, ctmrg_env0, mcf_env0;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    optimizer_alg,
    reuse_env,
    symmetrization,
    (finalize!) = (gauge_finalize!),
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
