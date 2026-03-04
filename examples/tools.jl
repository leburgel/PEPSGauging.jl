using Accessors
using TensorKit
using PEPSKit
using MPSKit
using VectorInterface
using CairoMakie
using OptimKit
using Zygote
using LinearAlgebra
using JLD2

using MatrixAlgebraKit: TruncationStrategy
using PEPSKit: hook_pullback
using PEPSGauging: generate_gauge_preserving_costfunction

function nogauge_energy_tracker(
        es0::Vector{T}, es1::Vector{T}, ngs::Vector{TR}, H::LocalOperator, env1::CTMRGEnv;
        boundary_alg = SimultaneousCTMRG(), save_iter = 10, fname = "result.jld2"
    ) where {T <: Number, TR <: Real}
    env1 = deepcopy(env1)
    extra_boundary_alg = @set boundary_alg.tol = boundary_alg.tol * 1.0e4 # relax this one a bit, only need the energy
    function finalize!(x, E, g, numiter)
        env0´, = leading_boundary(x[2], x[1], boundary_alg)
        e0 = MPSKit.expectation_value(x[1], H, env0´)
        env1´, = leading_boundary(env1, x[1], extra_boundary_alg)
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
    extra_boundary_alg = @set boundary_alg.tol = boundary_alg.tol * 1.0e4 # relax this one a bit, only need the energy
    function finalize!(x, E, g, numiter)
        peps´, = gauge_fix(x[1], gauge_alg) # gauge fix first
        env0´, = leading_boundary(x[2], peps´, boundary_alg)
        e0 = MPSKit.expectation_value(peps´, H, env0´)
        env1´, = leading_boundary(env1, peps´, extra_boundary_alg)
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

function test_peps_gradient(
        peps,
        boundary_env,
        gauge_env,
        H;
        gauge_alg = PEPSGauging.default_gauge_alg(),
        gauge_gradient_alg = nothing,
        svd_alg = PEPSGauging.default_svd_alg(),
        boundary_alg = PEPSGauging.default_boundary_alg(),
        boundary_gradient_alg = PEPSGauging.default_gradient_alg(),
        symmetrization = nothing,
        retract = PEPSGauging.gp_peps_retract,
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
        boundary_alg = PEPSGauging.default_boundary_alg(),
        gradient_alg = PEPSGauging.default_gradient_alg(),
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
        boundary_alg = PEPSGauging.default_boundary_alg(),
        gradient_alg = PEPSGauging.default_gradient_alg(),
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

function set_blas_threads(args::AbstractDict)
    nbt = args["blas_threads"]
    return set_blas_threads(nbt)
end
function set_blas_threads(nbt::Int)
    return if :MKL in filter((x) -> typeof(getfield(Main, x)) <: Module && x ≠ :Main, names(Main, imported = true))
        MKL.set_num_threads(nbt)
    else
        BLAS.set_num_threads(nbt)
    end
end

# IO

function update_peps_result(fname::String, x::Tuple{<:InfinitePEPS, <:CTMRGEnv})
    # get the histories
    fs, ngs, algs = jldopen(fname, "r") do file
        file["fs"], file["ngs"], file["algs"]
    end
    # combine these with the new tensors
    peps, env = x
    jldsave(fname; peps, env, fs, ngs, algs)
    return nothing
end

function generate_iterative_finalize(
        fname::String, (og_finalize!) = OptimKit._finalize!; frequency = 10, algs = (;)
    )
    # initialize histories
    fs = Float64[]
    ngs = Float64[]
    # initialize file if it does not exist
    if !isfile(fname)
        jldsave(fname; fs, ngs, algs)
    end
    # save every few iterations
    return function finalize!(x::Tuple, f, g, iter)
        push!(fs, f)
        push!(ngs, norm(g))
        x, f, g = og_finalize!(x, f, g, iter)
        if mod(iter, frequency) == 0
            # save
            @info "Saving checkpoint"
            peps, env = x
            jldsave(fname; peps, env, fs, ngs, algs)
        end
        return x, f, g
    end
end

function generate_reshuffling_finalize(
        boundary_alg, trunc::TruncationStrategy; frequency = 5, iters = 5, verbosity = 3
    )
    reshuffling_boundary_alg = @set boundary_alg.projector_alg.trunc = trunc
    @reset reshuffling_boundary_alg.maxiter = iters
    @reset reshuffling_boundary_alg.miniter = iters
    @reset reshuffling_boundary_alg.verbosity = verbosity
    function reshuffling_finalize!(x, f, g, iter)
        # reshuffle environment
        boundary_env = x[2]
        if mod(iter, frequency) == 0
            boundary_env, = leading_boundary(boundary_env, x[1], reshuffling_boundary_alg)
        end
        return (x[1], boundary_env, x[3:end]...), f, g
    end

    return reshuffling_finalize!
end

function load_result(fname::String, ::PEPSKit.LocalOperator)
    @info "Loading $(fname)"
    isfile(fname) || return nothing, nothing
    return jldopen(fname, "r") do file
        peps = read(file, "peps")
        isnothing(peps) && return nothing, nothing
        env = read(file, "env")
        return peps, env
    end
end
