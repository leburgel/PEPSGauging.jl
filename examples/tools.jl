using TensorKit
using PEPSKit
using MPSKit
using VectorInterface

function nogauge_energy_tracker(
        es0::Vector{T}, es1::Vector{T}, ngs::Vector{TR}, H::LocalOperator, env1::CTMRGEnv;
        boundary_alg = SimultaneousCTMRG(),
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
        return x, E, g, numiter
    end
    return finalize!
end

function gauge_energy_tracker(
        es0::Vector{T}, es1::Vector{T}, ngs::Vector{TR}, H::LocalOperator, env1::CTMRGEnv;
        boundary_alg = SimultaneousCTMRG(), gauge_alg = BeliefPropagation(),
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
