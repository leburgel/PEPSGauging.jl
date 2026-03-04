# attempt at imposing minial canonical form on single-site PEPS

#=
MCF gauge transform:

                   |
               exp(N_rc)
                   |
--exp(-E_r(c-1))--A_rc--exp(E_rc)--
                   |
               exp(-N_(r+1)c)
                   |

where N and E are Hermitian.
=#


@kwdef struct MCF
    tol::Real = 1.0e-10
    maxiter::Int = 500
    verbosity::Int = 1
end

const MCFEnv{T} = Tuple{T, T} where {T <: BondTensors}

function mcf_environment(psi::InfinitePEPS)
    Ns = map(eachcoordinate(psi)) do (r, c)
        N = randn(scalartype(psi), north_virtualspace(psi, r, c)' ← north_virtualspace(psi, r, c)')
        N = project_traceless_hermitian(N)
        return N / norm(N)
    end
    Es = map(eachcoordinate(psi)) do (r, c)
        E = randn(scalartype(psi), east_virtualspace(psi, r, c)' ← east_virtualspace(psi, r, c)')
        E = project_traceless_hermitian(E)
        return E / norm(E)
    end
    return Ns, Es
end
@non_differentiable mcf_environment(args...)

function PEPSKit.update!(env::MCFEnv{C}, env´::MCFEnv{C}) where {C}
    copy!.(env[1], env´[1])
    copy!.(env[2], env´[2])
    return env
end

function absorb_mcf_gauge_transform(psi::InfinitePEPS, env::MCFEnv)
    Ns, Es = env
    gauged_tensors = map(eachcoordinate(psi)) do (r, c)
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
        return A´
    end
    return InfinitePEPS(gauged_tensors)
end

function generate_mcf_costfun(psi::InfinitePEPS)
    function mcf_costfun((Ns, Es))
        f, gs = Zygote.withgradient((Ns, Es)) do (ns, es)
            # project hermitian
            ns = project_traceless_hermitian.(ns)
            es = project_traceless_hermitian.(es)
            # absorb gauge transform
            psi´ = absorb_mcf_gauge_transform(psi, (ns, es))
            return norm(psi´) # TODO: gradients of norm of unitcell?
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
    Ns0, Es0 = env
    # find the gauge transformation
    mcf_costfun = generate_mcf_costfun(psi)
    (Ns, Es), = optimize(
        mcf_costfun,
        (Ns0, Es0),
        OptimKit.LBFGS(
            32; gradtol = alg.tol, verbosity = alg.verbosity, maxiter = alg.maxiter
        ),
    )
    # apply it
    psi´ = absorb_mcf_gauge_transform(psi, (Ns, Es))
    # normalize each PEPS tensor individually afterwards
    psi´ = peps_normalize(psi´)
    return psi´, nothing, (Ns, Es) # HACK: match signature
end
