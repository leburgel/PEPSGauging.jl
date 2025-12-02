using Pkg: Pkg
Pkg.activate(@__DIR__)

using Revise

using Random
using TensorKit
using PEPSKit
using OptimKit
using PEPSKit: SUState, gauge_fix, compare_weights, peps_normalize

## SETUP

sd = 1234
unitcell = (2, 2)
P = ComplexSpace(2)
Vpeps = ComplexSpace(3)
Venv = ComplexSpace(20)
stype = ComplexF64
maxiter = 1000
tol = 1.0e-8
verbosity = 2

"""
A dummy Hamiltonian containing identity gates on all nearest neighbor bonds.
"""
function dummy_ham(elt::Type{<:Number}, lattice::Matrix{S}) where {S <: ElementarySpace}
    terms = []
    for site1 in CartesianIndices(lattice)
        r1, c1 = (mod1(x, N) for (x, N) in zip(site1.I, size(lattice)))
        for d in (CartesianIndex(1, 0), CartesianIndex(0, 1))
            site2 = site1 + d
            r2, c2 = (mod1(x, N) for (x, N) in zip(site2.I, size(lattice)))
            V1, V2 = lattice[r1, c1], lattice[r2, c2]
            h = TensorKit.id(elt, V1 ⊗ V2)
            push!(terms, (site1, site2) => h)
        end
    end
    return LocalOperator(lattice, terms...)
end

"""
PEPS gauge fixing using trivial simple update imaginary time evolution.
"""
function gauge_fix_su(
        peps::InfinitePEPS; maxiter::Int = 100, tol::Float64 = 1.0e-8, alg = SimpleUpdate()
    )
    H = dummy_ham(scalartype(peps), physicalspace(peps))
    wts0 = SUWeight(peps)
    ϵ = Inf
    # use default constructor to avoid calculation of exp(-H * 0)
    evolver = TimeEvolver(alg, 0.0, maxiter, H, SUState(0, 0.0, peps, wts0))
    for (i, (peps, wts, info)) in enumerate(evolver)
        ϵ = compare_weights(wts, wts0)
        if i % 10 == 0 || ϵ < tol
            @info "SU gauging step $i: ϵ = $ϵ."
            (ϵ < tol) && return peps, wts, ϵ
        end
        wts0 = deepcopy(wts)
    end
    return peps, wts0, ϵ
end


## MAIN

Random.seed!(sd)

H = heisenberg_XYZ(InfiniteSquare(unitcell...))

bp_alg = BeliefPropagation(; maxiter, tol, verbosity = verbosity + 1)
su_alg = SimpleUpdate(; trunc = FixedSpaceTruncation())
boundary_alg = SimultaneousCTMRG(; maxiter, tol, verbosity)
gradient_alg = LinSolver(; solver_alg = (; alg = :gmres), tol, maxiter, iterscheme = :fixed)

# make random PEPS
peps = peps_normalize(
    InfinitePEPS(randn, stype, fill(P, unitcell...), fill(Vpeps, unitcell...))
)

# gauge fix with BP
peps_bp, wts_bp, = gauge_fix(peps, bp_alg, BPEnv(ones, stype, peps)) # TODO: fix what I broke
normalize!.(wts_bp.data)
# peps_bp = peps_normalize(peps_bp)

# gauge fix with SU
peps_su, wts_su = gauge_fix_su(peps; maxiter, tol, alg = su_alg)
normalize!.(wts_su.data)
# peps_su = peps_normalize(peps_su)

# compare
@show norm(wts_bp.data - wts_su.data) / norm(wts_bp.data)

# contract all of the states
ctm_env0 = CTMRGEnv(randn, stype, peps, Venv)
ctm_env, = leading_boundary(ctm_env0, peps, boundary_alg)
ctm_env_bp, = leading_boundary(ctm_env0, peps_bp, boundary_alg)
ctm_env_su, = leading_boundary(ctm_env0, peps_su, boundary_alg)

# check network value and energy expectation value before and after gauging
n = network_value(peps, ctm_env)
n_bp = network_value(peps_bp, ctm_env_bp)
n_su = network_value(peps_su, ctm_env_su)
e = expectation_value(peps, H, ctm_env)
e_bp = expectation_value(peps_bp, H, ctm_env_bp)
e_su = expectation_value(peps_su, H, ctm_env_su)

@info "n = \t$n"
@info "n_bp = \t$n_bp"
@info "n_su = \t$n_su"
@info "e = \t$e"
@info "e_bp = \t$e_bp"
@info "e_su = \t$e_su"


## same thing after some LBFGS steps

optimizer_alg = LBFGS(32; maxiter = 20, verbosity = 3)
peps_opt, = fixedpoint(H, peps, ctm_env; optimizer_alg, gradient_alg, boundary_alg)

# gauge fix with BP
peps_opt_bp, wts_opt_bp, = gauge_fix(peps_opt, bp_alg, BPEnv(ones, stype, peps_opt))
normalize!.(wts_opt_bp.data)
# peps_opt_bp = peps_normalize(peps_opt_bp)

# gauge fix with SU
peps_opt_su, wts_opt_su = gauge_fix_su(peps_opt; maxiter, tol, alg = su_alg)
normalize!.(wts_opt_su.data)
# peps_opt_su = peps_normalize(peps_opt_su) # NOTE: this one has a really messed up norm...

# compare
@show norm(wts_opt_bp.data - wts_opt_su.data)

# contract all of the states
ctm_env0 = CTMRGEnv(randn, stype, peps_opt, Venv)
ctm_env = leading_boundary(ctm_env0, peps_opt, boundary_alg)
ctm_env_bp = leading_boundary(ctm_env0, peps_opt_bp, boundary_alg)
ctm_env_su = leading_boundary(ctm_env0, peps_opt_su, boundary_alg)

# check network value and energy expectation value before and after gauging
n_opt = network_value(peps_opt, ctm_env)
n_opt_bp = network_value(peps_opt_bp, ctm_env_bp)
n_opt_su = network_value(peps_opt_su, ctm_env_su)
e_opt = expectation_value(peps_opt, H, ctm_env)
e_opt_bp = expectation_value(peps_opt_bp, H, ctm_env_bp)
e_opt_su = expectation_value(peps_opt_su, H, ctm_env_su)

@info "n_opt = \t$n_opt"
@info "n_opt_bp = \t$n_opt_bp"
@info "n_opt_su = \t$n_opt_su"
@info "e_opt = \t$e_opt"
@info "e_opt_bp = \t$e_opt_bp"
@info "e_opt_su = \t$e_opt_su"

nothing
