using TensorKit
using PEPSKit
using MPSKit
using VectorInterface
using CairoMakie
using OptimKit
using Zygote
using BPAD: generate_gauge_preserving_costfunction

using PEPSKit: hook_pullback

function nogauge_energy_tracker(
        es0::Vector{T}, es1::Vector{T}, ngs::Vector{TR}, H::LocalOperator, env1::CTMRGEnv;
        boundary_alg = SimultaneousCTMRG(), save_iter = 10, fname = "result.jld2"
    ) where {T <: Number, TR <: Real}
    env1 = deepcopy(env1)
    function finalize!(x, E, g, numiter)
        env0´, = leading_boundary(x[2], x[1], boundary_alg) # make sure this is an environment
        e0 = MPSKit.expectation_value(x[1], H, env0´)
        env1´, = leading_boundary(env1, x[1], boundary_alg) # make sure this is an environment
        e1 = MPSKit.expectation_value(x[1], H, env1´)
        PEPSKit.update!(env1, env1´)
        ng = sqrt(PEPSKit.real_inner(x, g, g))
        push!(es0, e0)
        push!(es1, e1)
        push!(ngs, ng)
        if mod(numiter, save_iter) == 0
            jldsave(
                fname;
                peps = x[1],
                env = x[2],
                es0,
                es1,
                ngs,
            )
        end
        return x, E, g, numiter
    end
    return finalize!
end

function gauge_energy_tracker(
        es0::Vector{T}, es1::Vector{T}, ngs::Vector{TR}, H::LocalOperator, env1::CTMRGEnv;
        boundary_alg = SimultaneousCTMRG(), gauge_alg = BeliefPropagation(),
        save_iter = 10, fname = "result.jld2"
    ) where {T <: Number, TR <: Real}
    env1 = deepcopy(env1)
    function finalize!(x, E, g, numiter)
        peps´, = gauge_fix(x[1], gauge_alg) # gauge fix first
        env0´, = leading_boundary(x[2], peps´, boundary_alg) # make sure this is an environment
        e0 = MPSKit.expectation_value(peps´, H, env0´)
        env1´, = leading_boundary(env1, peps´, boundary_alg) # make sure this is an environment
        e1 = MPSKit.expectation_value(peps´, H, env1´)
        PEPSKit.update!(env1, env1´)
        ng = sqrt(PEPSKit.real_inner(x, g, g))
        push!(es0, e0)
        push!(es1, e1)
        push!(ngs, ng)
        if mod(numiter, save_iter) == 0
            jldsave(
                fname;
                peps = x[1],
                env = x[2],
                es0,
                es1,
                ngs,
            )
        end
        return x, E, g, numiter
    end
    return finalize!
end

function generate_heisenberg_filename(
        D,
        chi,
        chi1,
        gauge = false,
        symmetrization = nothing,
        Jx = -1.0,
        Jy = 1.0,
        Jz = -1.0,
    )
    return "heisenberg_XYZ_D_$(D)_chi_$(chi)_chi1_$(chi1)_gauge_$(gauge)_symm_$(symmetrization)_Jx_$(Jx)_Jy_$(Jy)_Jz_$(Jz).jld2"
end

## Gradient testing things

# function gp_peps_fg(
#         H;
#         gauge_alg = BPAD.default_gauge_alg(),
#         gauge_gradient_alg = nothing,
#         svd_alg = BPAD.default_svd_alg(),
#         boundary_alg = BPAD.default_boundary_alg(),
#         boundary_gradient_alg = BPAD.default_gradient_alg(),
#         symmetrization = nothing,
#     )
#     function fg((peps, boundary_env, gauge_env))
#         E, gs = withgradient(peps) do ψ
#             ψg, _, gauge_env´ = hook_pullback(
#                 gauge_fix, ψ, gauge_alg, gauge_env, svd_alg; alg_rrule = gauge_gradient_alg,
#             ) # the bamboozle
#             boundary_env′, = hook_pullback(
#                 leading_boundary, boundary_env, ψg, boundary_alg;
#                 alg_rrule = boundary_gradient_alg,
#             )
#             return cost_function(ψg, boundary_env′, H)
#         end
#         g = only(gs)
#         symmetrize!(g, symmetrization)
#         return E, g
#     end
#     return fg
# end

function test_peps_gradient(
        peps,
        boundary_env,
        gauge_env,
        H;
        gauge_alg = BPAD.default_gauge_alg(),
        gauge_gradient_alg = nothing,
        svd_alg = BPAD.default_svd_alg(),
        boundary_alg = BPAD.default_boundary_alg(),
        boundary_gradient_alg = BPAD.default_gradient_alg(),
        symmetrization = nothing,
        retract = BPAD.gp_peps_retract,
        inner = PEPSKit.real_inner,
        steps = LinRange(-0.01, 0.1, 12),
        doplot = false,
    )
    fg = generate_gauge_preserving_costfunction(
        H,
        gauge_alg,
        gauge_gradient_alg,
        svd_alg, boundary_alg,
        boundary_gradient_alg,
        symmetrization,
    )
    f0, g0 = fg((peps, boundary_env, gauge_env))
    ng0 = inner(peps, g0, g0)
    dir = -1 * g0 # descent direction = negative gradient direction
    alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
        fg,
        (peps, boundary_env, gauge_env),
        dir;
        alpha = steps,
        retract,
        inner,
    )
    fig = Figure()
    if doplot
        # plot both of them
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = L"\alpha", ylabel = L"f")
        scatterlines!(ax, alphas, fs, label = L"f(\vec{x}-\alpha\vec{\nabla} f)")
        scatterlines!(ax, alphas, f0 .- alphas .* ng0, label = L"f(\vec{x})-\alpha||\vec{\nabla} f||^2")
        hlines!(ax, [f0], linestyle = :dash, color = :black, label = L"f(\vec{x})")
        axislegend(ax; position = :rt)
    end

    return f0, g0, alphas, fs, dfs1, dfs2, fig
end

# TODO: should be able to get this directly from PEPSKit somehow...
function regular_peps_fg(
        H;
        boundary_alg = BPAD.default_boundary_alg(),
        gradient_alg = BPAD.default_gradient_alg(),
        symmetrization = nothing,
    )
    function fg((peps, env))
        E, gs = withgradient(peps) do ψ
            env′, = hook_pullback(
                leading_boundary, env, ψ, boundary_alg; alg_rrule = gradient_alg,
            )
            return cost_function(ψ, env′, H)
        end
        g = only(gs)
        symmetrize!(g, symmetrization)
        return E, g
    end
    return fg
end

function test_regular_peps_gradient(
        peps,
        env,
        H;
        boundary_alg = BPAD.default_boundary_alg(),
        gradient_alg = BPAD.default_gradient_alg(),
        symmetrization = nothing,
        retract = PEPSKit.peps_retract,
        inner = PEPSKit.real_inner,
        steps = LinRange(-0.01, 0.1, 12),
        doplot = false,
    )
    fg = regular_peps_fg(
        H;
        boundary_alg,
        gradient_alg,
        symmetrization,
    )
    f0, g0 = fg((peps, env))
    ng0 = inner(peps, g0, g0)
    dir = -1 * g0 # descent direction = negative gradient direction
    alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
        fg,
        (peps, env),
        dir;
        alpha = steps,
        retract,
        inner,
    )
    fig = Figure()
    if doplot
        # plot both of them
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = L"\alpha", ylabel = L"f")
        scatterlines!(ax, alphas, fs, label = L"f(\vec{x}-\alpha\vec{\nabla} f)")
        scatterlines!(ax, alphas, f0 .- alphas .* ng0, label = L"f(\vec{x})-\alpha||\vec{\nabla} f||^2")
        hlines!(ax, [f0], linestyle = :dash, color = :black, label = L"f(\vec{x})")
        axislegend(ax; position = :rt)
    end

    return f0, g0, alphas, fs, dfs1, dfs2, fig
end
