# attempt at imposing minial canonical form on single-site PEPS

#=
MCF gauge transform:

                   |
               exp(N_rc) * exp(α_rc)
                   |
--exp(-E_r(c-1))--A_rc--exp(E_rc)--
                   |
               exp(-N_(r+1)c)
                   |

where N_rc and E_rc are Hermitian, and α_rc = Σ_j αs[(r, c), j] - Σ_j αs[i, (r, c)] is the
overall scaling factor for site (r, c).
=#


@kwdef struct MCF
    tol::Real = 1.0e-10
    maxiter::Int = 500
    verbosity::Int = 1
end

const MCFEnv{C, A} = Tuple{C, C, A} where {C <: BondTensors, A <: AbstractMatrix{<:Real}}

function mcf_environment(psi::InfinitePEPS)
    # north (and south) bond gauge tensors
    Ns = map(eachcoordinate(psi)) do (r, c)
        N = randn(scalartype(psi), north_virtualspace(psi, r, c)' ← north_virtualspace(psi, r, c)')
        N = project_traceless_hermitian(N)
        return N / norm(N)
    end
    # east (and west) bond gauge tensors
    Es = map(eachcoordinate(psi)) do (r, c)
        E = randn(scalartype(psi), east_virtualspace(psi, r, c)' ← east_virtualspace(psi, r, c)')
        E = project_traceless_hermitian(E)
        return E / norm(E)
    end
    # overall scaling factors, one for every unique pair of sites (i, j) = ((r, c), (r′, c′))
    αs = triu(randn(real(scalartype(psi)), length(psi), length(psi)), 1)
    return Ns, Es, αs
end
@non_differentiable mcf_environment(args...)

function PEPSKit.update!(env::MCFEnv{C, A}, env´::MCFEnv{C, A}) where {C, A}
    copy!.(env[1], env´[1])
    copy!.(env[2], env´[2])
    copy!(env[3], env´[3])
    return env
end

function absorb_mcf_gauge_transform(psi::InfinitePEPS, env::MCFEnv)
    Ns, Es, α = env
    gauged_tensors = map(enumerate(eachcoordinate(psi))) do (i, (r, c))
        # apply appropriate bond gauge tensors
        eN = exp(Ns[r, c])
        eE = exp(Es[r, c])
        eS = exp(-Ns[_next(r, end), c])
        eW = exp(-Es[r, _prev(c, end)])
        @tensor A´[d; No Eo So Wo] :=
            psi[r, c][d; N E S W] *
            eN[N; No] *
            eE[E; Eo] *
            eS[So; S] *
            eW[Wo; W]
        # rescale the resulting PEPS tensor according to the overall scaling factor for this site
        A´ *= exp(sum(α[i, :]) - sum(α[:, i]))
        return A´
    end
    return InfinitePEPS(gauged_tensors)
end

function _project_input(env::MCFEnv)
    Ns = map(project_traceless_hermitian, env[1])
    Es = map(project_traceless_hermitian, env[2])
    αs = triu(real(env[3]), 1)
    return Ns, Es, αs
end

function generate_mcf_costfun(psi::InfinitePEPS)
    function mcf_costfun((Ns, Es, αs))
        f, gs = Zygote.withgradient((Ns, Es, αs)) do env
            # project out anything that shouldn't be there
            ns, es, α = _project_input(env)
            # absorb gauge transform
            psi´ = absorb_mcf_gauge_transform(psi, (ns, es, α))
            return norm(psi´)
        end
        g = only(gs)
        return f, g
    end
    return mcf_costfun
end

function PEPSKit.gauge_fix(
        psi::InfinitePEPS,
        alg::MCF,
        env::MCFEnv = mcf_environment(psi),
        svd_alg = nothing, # HACK: match signature
    )
    Ns0, Es0, αs0 = env
    # find the gauge transformation
    mcf_costfun = generate_mcf_costfun(psi)
    (Ns, Es, αs), = optimize(
        mcf_costfun,
        (Ns0, Es0, αs0),
        OptimKit.LBFGS(
            32; gradtol = alg.tol, verbosity = alg.verbosity, maxiter = alg.maxiter
        ),
    )
    # apply it
    psi´ = absorb_mcf_gauge_transform(psi, (Ns, Es, αs))
    # global rescaling to fix tensor norm of each tensor
    psi´ /= (norm(psi´) / sqrt(length(psi´)))
    return psi´, nothing, (Ns, Es, αs) # HACK: match signature
end
