# gauge-preserving PEPS optimization

default_gauge_alg() = MCF(;
    tol = 1.0e-6,
    maxiter = 10,
    verbosity = 2,
)

default_boundary_alg() = SimultaneousCTMRG(;
    alg = :simultaneous,
    tol = PEPSKit.Defaults.ctmrg_tol,
    maxiter = PEPSKit.Defaults.ctmrg_maxiter,
    miniter = PEPSKit.Defaults.ctmrg_miniter,
    verbosity = PEPSKit.Defaults.ctmrg_verbosity,
)

default_gradient_alg() = LinSolver(;
    solver_alg = KrylovKit.GMRES(;
        tol = PEPSKit.Defaults.fpgrad_tol,
        maxiter = PEPSKit.Defaults.fpgrad_maxiter,
        verbosity = PEPSKit.Defaults.fpgrad_verbosity,
        krylovdim = 100,
    ),
    iterscheme = :fixed,
)

default_linesearch_alg() = HagerZhangLineSearch(;
    c₁ = 1.0e-4,
    c₂ = 1 - 1.0e-4,
    maxiter = PEPSKit.Defaults.ls_maxiter,
    maxfg = PEPSKit.Defaults.ls_maxfg,
)

default_optimizer_alg() = LBFGS(
    32;
    gradtol = PEPSKit.Defaults.optimizer_tol,
    verbosity = PEPSKit.Defaults.optimizer_verbosity,
    maxiter = PEPSKit.Defaults.optimizer_maxiter,
    linesearch = default_linesearch_alg(),
)

default_svd_alg() = SVDAdjoint(;
    fwd_alg = (; alg = :sdd),
    rrule_alg = (; alg = :full),
)

const GaugeEnv = Union{BPEnv, MCFEnv}

function generate_gauge_preserving_costfunction(
        operator::LocalOperator,
        gauge_alg::Union{BeliefPropagation, MCF}, # these gauge using only the PEPS
        gauge_gradient_alg,
        svd_alg,
        boundary_alg,
        boundary_gradient_alg,
        symmetrization,
        gradnorms_unitcell = [],
        times = [];
        reuse_env = true,
    )
    function gp_costfun((peps, boundary_env, gauge_env))
        start_time = time_ns()
        E, gs = withgradient(peps) do ψ
            ψg, _, gauge_env´ = hook_pullback(
                gauge_fix, ψ, gauge_alg, gauge_env, svd_alg; alg_rrule = gauge_gradient_alg,
            ) # the bamboozle
            boundary_env′, = hook_pullback(
                leading_boundary, boundary_env, ψg, boundary_alg;
                alg_rrule = boundary_gradient_alg,
            )
            ignore_derivatives() do
                reuse_env && (update!(boundary_env, boundary_env′); update!(gauge_env, gauge_env´))
            end
            return cost_function(ψg, boundary_env′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        symmetrize!(g, symmetrization)
        push!(gradnorms_unitcell, norm.(g.A))
        push!(times, (time_ns() - start_time) * 1.0e-9)
        return E, g
    end
    return gp_costfun
end

# just because the original one is not general enough...
function gauge_preserving_fixedpoint(
        operator,
        peps0::InfinitePEPS,
        boundary_env0::CTMRGEnv,
        gauge_env0::GaugeEnv; # the bamboozle
        gauge_alg = default_gauge_alg(),
        gauge_gradient_alg = nothing,
        svd_alg = default_svd_alg(),
        boundary_alg = default_boundary_alg(),
        boundary_gradient_alg = default_gradient_alg(),
        optimizer_alg = default_optimizer_alg(),
        reuse_env = PEPSKit.Defaults.reuse_env,
        symmetrization = nothing,
        (finalize!) = OptimKit._finalize!,
    )

    # setup retract and finalize! for symmetrization
    if isnothing(symmetrization)
        retract = gp_peps_retract
    else
        retract, finalize! = gp_symmetrize_retract_and_finalize!(
            symmetrization, gp_peps_retract, finalize!
        )
    end

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps0)), real(scalartype(boundary_env0)))
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # normalize the initial guess
    peps0 = peps_normalize(peps0)

    # optimize operator cost function
    gp_costfun = generate_gauge_preserving_costfunction(
        operator,
        gauge_alg,
        gauge_gradient_alg,
        svd_alg,
        boundary_alg,
        boundary_gradient_alg,
        symmetrization,
        gradnorms_unitcell,
        times;
        reuse_env,
    )

    (peps_final, env_final), cost_final, ∂cost, numfg, convergence_history = optimize(
        gp_costfun, (peps0, boundary_env0, gauge_env0), optimizer_alg;
        retract, inner = real_inner, finalize!, (transport!) = (gp_peps_transport!),
    )

    info = (;
        last_gradient = ∂cost,
        fg_evaluations = numfg,
        costs = convergence_history[:, 1],
        gradnorms = convergence_history[:, 2],
        gradnorms_unitcell,
        times,
    )
    return peps_final, env_final, cost_final, info
end

# some boilerplate; who the hell hardcodes tuple lengths in the first place...
function gp_peps_retract((peps, boundary_env, gauge_env), η, α)
    (peps´, eboundary_env´), ξ = PEPSKit.peps_retract((peps, boundary_env), η, α)
    gauge_env´ = deepcopy(gauge_env)
    return (peps´, eboundary_env´, gauge_env´), ξ
end

function gp_peps_transport!(
        ξ,
        (peps, boundary_env, gauge_env),
        η,
        α,
        (peps´, boundary_env´, gauge_env´),
    )
    return PEPSKit.peps_transport!(
        ξ, (peps, boundary_env), η, α, (peps´, boundary_env´)
    )
end

function gp_symmetrize_retract_and_finalize!(
        symm::SymmetrizationStyle, retract = gp_peps_retract, (finalize!) = OptimKit._finalize!
    )
    function symmetrize_then_finalize!(x, E, grad, numiter)
        # symmetrize the gradient
        grad_symm = symmetrize!(grad, symm)
        # then finalize
        return finalize!(x, E, grad_symm, numiter)
    end
    function retract_then_symmetrize(x, η, α)
        # retract
        x´, ξ = retract(x, η, α)
        # symmetrize retracted point and directional derivative
        symmetrize!(first(x´), symm)
        symmetrize!(ξ, symm)
        return x´, ξ
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end
