"""
Test out MCF-preserving optimization on J1-J2-J3 model with SU(2) symmetry.
"""

using Revise

using Random
using PEPSKit
using TensorKit
using OptimKit
using KrylovKit
using JLD2
using MPSKitModels: S_exchange
using PEPSGauging

include(joinpath(pwd(), "tools.jl"))

using MPSKit: randomize!
using PEPSKit: peps_normalize, gauge_fix, unitcell
using PEPSGauging: MCF, mcf_environment

sd = 12345

set_blas_threads(16)

# SETUP
# -----

lattice = InfiniteSquare(2, 2)

N = 2
D = 8
χ = 100

I = fℤ₂ #symmetry sector
S = SU2Irrep

# define physical and virtual spaces
P = Vect[S](1 // 2 => 1);
V_int = Vect[S](0 => 1)
V_halfint = Vect[S](1 / 2 => 1)
V_env = Vect[S](0 => 10, 1 / 2 => 10);

# set up truncation schemes
trunc_peps = truncrank(D)
trunc_env = truncrank(χ)

# gauge algorithm settings
gauge_tol = 1.0e-9
gauge_maxiter = 500
gauge_verbosity = 2

# contraction algorithm settings
boundary_tol = 1.0e-9
boundary_maxiter = 150
boundary_verbosity = 2

# gradient algorithm settings
gradient_tol = 5.0e-8
gradient_maxiter = 5
gradient_krylovdim = 500
gradient_verbosity = 2

# linesearch algorithm settingsq
linesearch_c1 = 1.0e-4
linesearch_c2 = 1 - 1.0e-4
linesearch_maxiter = 4
linesearch_verbosity = 3
linesearch_maxfg = 4

# optimizer algorithm settings
optim_tol = 1.0e-5
optim_verbosity = 3
optim_maxiter = 400

# time evolution algorithms settings
time_evolve_dts = [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8] #, 1.0e-10]
time_evolve_tols = [1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12] #, 1.0e-12]
time_evolve_maxiter = 3000
time_evolve_check_interval = 1000
time_evolve_bipartite = false

# fixedpoint settings
reuse_env = true
symmetrization = nothing


# AUXILIARY FUNCTIONS
# -------------------

function naichao_hamiltonian(lattice::InfiniteSquare; kwargs...)
    return naichao_hamiltonian(ComplexF64, SU2Irrep, lattice; kwargs...)
end

function naichao_hamiltonian_for_simple_update(lattice::InfiniteSquare; kwargs...)
    return naichao_hamiltonian_for_simple_update(ComplexF64, SU2Irrep, lattice; kwargs...)
end

function naichao_hamiltonian(
        T::Type{<:Complex}, S::Type{<:Sector}, lattice::InfiniteSquare;
        J1 = 1.0, J2 = 1.0, J3 = 0.5, spin = 1 // 2
    )
    term_NN = rmul!(S_exchange(T, S; spin = spin), J2)
    term_diag = rmul!(S_exchange(T, S; spin = spin), J1)
    term_NNN = rmul!(S_exchange(T, S; spin = spin), J3)
    spaces = fill(domain(term_NN)[1], (lattice.Nrows, lattice.Ncols))
    return PEPSKit.LocalOperator(
        spaces,
        (neighbor => term_NN for neighbor in PEPSKit.nearest_neighbours(lattice))...,
        (neighbor => term_diag for neighbor in diagonal_ourcase(lattice))...,
        (neighbor => term_NNN for neighbor in nnn_naichao_neighbor(lattice))...,
    )
end

function naichao_hamiltonian_for_simple_update(
        T::Type{<:Complex}, S::Type{<:Sector}, lattice::InfiniteSquare;
        J1 = 1.0, J2 = 1.0, J3 = 0.5, spin = 1 // 2
    )
    term_NN = rmul!(S_exchange(T, S; spin = spin), J2)
    spaces = fill(domain(term_NN)[1], (lattice.Nrows, lattice.Ncols))
    return PEPSKit.LocalOperator(
        spaces,
        (neighbor => term_NN for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    )
end

function nnn_naichao_neighbor(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex, CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(2, 0)))
            push!(neighbors, (idx, idx + CartesianIndex(0, 2)))
        elseif iseven(idx[1]) && iseven(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(2, 0)))
            push!(neighbors, (idx, idx + CartesianIndex(0, 2)))
        end
    end
    return neighbors
end

function diagonal_ourcase(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex, CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)

        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(1, 1)))
            push!(neighbors, (idx, idx + CartesianIndex(1, -1)))
        elseif iseven(idx[1]) && iseven(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(1, 1)))
            push!(neighbors, (idx, idx + CartesianIndex(1, -1)))
        end

    end
    return neighbors
end

# INSTANTIATE ALGORITHMS
# ----------------------

gauge_alg = MCF(;
    tol = gauge_tol,
    maxiter = gauge_maxiter,
    verbosity = gauge_verbosity,
)

gauge_gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = gradient_tol,
        maxiter = gradient_maxiter,
        verbosity = gradient_verbosity,
        krylovdim = gradient_krylovdim,
    ),
    iterscheme = :characteristic,
)

svd_alg = nothing # DUMMY

boundary_alg = SimultaneousCTMRG(;
    alg = :simultaneous,
    tol = boundary_tol,
    verbosity = boundary_verbosity,
    maxiter = boundary_maxiter,
    trunc = trunc_env,
)

boundary_gradient_alg = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = gradient_tol,
        maxiter = gradient_maxiter,
        verbosity = gradient_verbosity,
        krylovdim = gradient_krylovdim,
    ),
    iterscheme = :fixed,
)

lineasearch_alg = HagerZhangLineSearch(;
    c₁ = linesearch_c1,
    c₂ = linesearch_c2,
    maxiter = linesearch_maxiter,
    verbosity = linesearch_verbosity,
    maxfg = linesearch_maxfg,
)

optimizer_alg = LBFGS(
    32; gradtol = optim_tol, verbosity = optim_verbosity, maxiter = optim_maxiter,
    linesearch = lineasearch_alg,
)

time_evolve_alg = SimpleUpdate(; trunc = trunc_peps, bipartite = time_evolve_bipartite)


# MAIN
# ----

Random.seed!(sd)

# initialize Hamiltonian
H_total = naichao_hamiltonian(lattice)
H_simple_update = naichao_hamiltonian_for_simple_update(lattice)

# initialize state via imaginary time evolution
pA = TensorMap(randn, ComplexF64, P ← V_int ⊗ V_halfint ⊗ V_halfint' ⊗ V_halfint')
pB = TensorMap(randn, ComplexF64, P ← V_int ⊗ V_halfint ⊗ V_halfint' ⊗ V_halfint')
pC = TensorMap(randn, ComplexF64, P ← V_halfint ⊗ V_halfint ⊗ V_int' ⊗ V_halfint')
pD = TensorMap(randn, ComplexF64, P ← V_halfint ⊗ V_halfint ⊗ V_int' ⊗ V_halfint')

psi0 = InfinitePEPS([pA pB; pC pD])
wts = SUWeight(psi0)

for (dt, tol) in zip(time_evolve_dts, time_evolve_tols)
    global psi0, wts, = PEPSKit.time_evolve(
        psi0, H_simple_update, dt, time_evolve_maxiter, time_evolve_alg, wts;
        tol, check_interval = time_evolve_check_interval,
    )
end

# randomize simple update result to obtain starting state
@show space.(unitcell(psi0))
randomize!.(unitcell(psi0))

# set up custom finalize! function to save data and periodically reshuffle the environment virtual space
reshuffling_finalize! = generate_reshuffling_finalize(
    boundary_alg,
    truncrank(χ);
    frequency = 5,
    iters = 10,
    verbosity = 3,
)
fname = joinpath(pwd(), "results", "glebs_example.jld2")
custom_finalize! = generate_iterative_finalize(fname, reshuffling_finalize!; frequency = 1)

# gauge fix and normalize
psi0, _, mcf_env0 = gauge_fix(psi0, gauge_alg)
psi0 = peps_normalize(psi0)

# initialize the environment
env0 = CTMRGEnv(psi0, V_env)
for χ_trunc in 40:20:χ
    ctm_alg_1 = SimultaneousCTMRG(;
        maxiter = 10, verbosity = 3, trunc = truncrank(χ_trunc),
    )
    global env0, = leading_boundary(env0, psi0, ctm_alg_1)
end
env0, = leading_boundary(env0, psi0, boundary_alg)

# optimize
psi, ctmrg_env, E, = gauge_preserving_fixedpoint(
    H_total, psi0, env0, mcf_env0;
    gauge_alg,
    gauge_gradient_alg,
    svd_alg,
    boundary_alg,
    boundary_gradient_alg,
    optimizer_alg,
    reuse_env,
    (finalize!) = (custom_finalize!),
)

# update final result
update_peps_result(fname, (psi, ctmrg_env))
