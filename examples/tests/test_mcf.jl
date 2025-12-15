"""
Test MCF-based optimization
"""

using Revise

using Test
using Random
using Accessors
using PEPSKit
using TensorKit
using OptimKit
using KrylovKit
using Zygote
using JLD2
using PEPSGauging

using PEPSKit: peps_normalize, BeliefPropagation, gauge_fix
using PEPSGauging: MCF, mcf_environment

include(joinpath(pwd(), "tools.jl"))

sd = 12345678

## Set up test
# ------------

P = ComplexSpace(2)
Vpeps = ComplexSpace(3)
Venv = ComplexSpace(20)

symmetrization = nothing # RotateReflect()
reuse_env = true

Jx = -1.0
Jy = 1.0
Jz = -1.0
H = heisenberg_XYZ(InfiniteSquare(); Jx, Jy, Jz)

gauge_tol = 1.0e-10
gauge_maxiter = 500
gauge_verbosity = 2

boundary_tol = 1.0e-10
boundary_maxiter = 500
boundary_verbosity = 2

fpgrad_tol = 1.0e-8
fpgrad_verbosity = 2

optim_tol = 1.0e-5
optim_verbosity = 3
optim_maxiter = 600 # for full Heisenberg model benchmark
optim_initial_step = 1.0

gauge_alg = MCF(;
    tol = gauge_tol,
    maxiter = gauge_maxiter,
    verbosity = gauge_verbosity,
)

gauge_gradient_alg = Val(:constant_bonds)

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

ls_alg = HagerZhangLineSearch(;
    c₁ = 1.0e-4,
    maxiter = 10,
    maxfg = 10,
)

optimizer_alg = LBFGS(
    32;
    gradtol = optim_tol,
    verbosity = optim_verbosity,
    maxiter = optim_maxiter,
    linesearch = ls_alg,
    initial_step = optim_initial_step,
)

#
# Check gauging: network value and energy
#

# seems to be working fine
peps0 = peps_normalize(InfinitePEPS(P, Vpeps))
ctmrg_env0, = leading_boundary(CTMRGEnv(peps0, Venv), peps0, boundary_alg)
peps0g, weights, mcf_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0g, = leading_boundary(CTMRGEnv(peps0, Venv), peps0g, boundary_alg)

n0 = network_value(peps0, ctmrg_env0)
n0g = network_value(peps0g, ctmrg_env0g)
e0 = expectation_value(peps0, H, ctmrg_env0)
e0g = expectation_value(peps0g, H, ctmrg_env0g)

@info "n0 = \t$n0"
@info "n0g = \t$n0g"
@info "e0 = \t$e0"
@info "e0g = \t$e0g"

#
# Check optimization: do 10 iterations and save
#

dummy_optimizer_alg = @set optimizer_alg.maxiter = 20

peps, ctmrg_env, E, = gauge_preserving_fixedpoint(
    H, peps0g, ctmrg_env0g, mcf_env0;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    optimizer_alg = dummy_optimizer_alg,
    reuse_env,
    symmetrization,
)

jldsave("mcf_peps_20_iter.jld2"; peps, ctmrg_env, H)

#
# Check gradients explicitly
#

steps_plot = LinRange(-0.05, 0.5, 21) # overview

Random.seed!(sd)
peps0 = peps_normalize(InfinitePEPS(P, Vpeps))

# contract, gauge and contract again
ctmrg_env0, = leading_boundary(CTMRGEnv(peps0, Venv), peps0, boundary_alg)
peps0g, weights, mcf_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0g, = leading_boundary(CTMRGEnv(peps0g, Venv), peps0g, boundary_alg)

## Test on random PEPS

# test the gradient starting from the ungauged PEPS
_, g0_before_gauge, _, _, _, _, fig_before_nogauge = test_peps_gradient(
    peps0, ctmrg_env0, mcf_env0, H;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)
@info "Gradient norm before optimization, ungauged PEPS: $(PEPSKit.real_inner(peps0, g0_before_nogauge, g0_before_nogauge))"


# test the gradient starting from the gauged PEPS
_, g0_before_gauge, _, _, _, _, fig_before_gauge = test_peps_gradient(
    peps0g, ctmrg_env0g, mcf_env0, H;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)
@info "Gradient norm before optimization, ungauged PEPS: $(PEPSKit.real_inner(peps0, g0_before_nogauge, g0_before_nogauge))"

## Test on partially optimized PEPS

data = load("mcf_peps_20_iter.jld2")
peps = data["peps"]
ctmrg_env = data["ctmrg_env"]
H = data["H"]

pepsg, _, mcf_env = gauge_fix(peps, gauge_alg)
ctmrg_envg, = leading_boundary(CTMRGEnv(pepsg, Venv), pepsg, boundary_alg)

# test the gradient starting from the ungauged PEPS
_, _, _, _, _, _, fig_after_nogauge = test_peps_gradient(
    peps, ctmrg_env, mcf_env, H;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

# test the gradient starting from the gauged PEPS
_, _, _, _, _, _, fig_after_gauge = test_peps_gradient(
    pepsg, ctmrg_envg, mcf_env, H;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    symmetrization,
    steps = steps_plot,
    doplot = true,
)

save("mcf_before_nogauge.png", fig_before_nogauge)
save("mcf_before_gauge.png", fig_before_gauge)
save("mcf_after_nogauge.png", fig_after_nogauge)
save("mcf_after_gauge.png", fig_after_gauge)

#
# Check optimization: track energies and gradients across many iterations
#

Random.seed!(sd)

T = ComplexF64
D = dim(Vpeps)
chi = dim(Venv)
chi1 = 50

Venv1 = ComplexSpace(chi1)
es0 = T[]
es1 = T[]
ngs = real(T)[]

peps0 = peps_normalize(symmetrize!(InfinitePEPS(randn, T, P, Vpeps), symmetrization))
peps0, weights, mcf_env0 = gauge_fix(peps0, gauge_alg)
ctmrg_env0, = leading_boundary(CTMRGEnv(randn, T, peps0, Venv), peps0, boundary_alg)
ctmrg_env1 = CTMRGEnv(randn, T, peps0, Venv1)

gauge_finalize! = gauge_energy_tracker(es0, es1, ngs, H, ctmrg_env1; boundary_alg, gauge_alg)

fname = joinpath(
    pwd(),
    "results",
    generate_heisenberg_filename(D, chi, chi1, "mcf", symmetrization, Jx, Jy, Jz),
)

peps, ctmrg_env, E, = gauge_preserving_fixedpoint(
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
