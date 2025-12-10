#
# Derivative of BP gauge fixing
#

function _rrule(
        gradmode::LinSolver{:characteristic},
        config::RuleConfig,
        ::typeof(gauge_fix),
        state::InfinitePEPS,
        alg::BeliefPropagation,
        env0::BPEnv,
    )

    throw(ArgumentError("BP gauging fixed-point gradient not yet implemented"))
end


#
# Derivative of MCF gauge fixing
#

# the hacked version
function _rrule(
        gradmode::Val{:constant_bonds},
        config::RuleConfig,
        ::typeof(gauge_fix),
        state::InfinitePEPS,
        alg::MCF,
        env0::MCFEnv,
        svd_alg::Nothing,
    )

    state_gauged, weights, mcf_env = gauge_fix(state, alg, env0, svd_alg)
    _, absorb_pb = pullback(absorb_mcf_gauge_transform, state, mcf_env)

    function gaugefix_pullback(Δout_)
        Δstate_gauged, = unthunk.(Δout_)
        Δstate, _ = absorb_pb(Δstate_gauged)
        return NoTangent(), Δstate, NoTangent(), NoTangent(), NoTangent()
    end

    return (state_gauged, weights, mcf_env), gaugefix_pullback
end

# the 'proper' fixed-point approach, see Fig. 8 of https://arxiv.org/abs/2209.14358
function _rrule(
        gradmode::LinSolver{:characteristic},
        config::RuleConfig,
        ::typeof(gauge_fix),
        state::InfinitePEPS,
        alg::MCF,
        env0::MCFEnv,
        svd_alg::Nothing,
    )

    state_gauged, weights, mcf_env = gauge_fix(state, alg, env0, svd_alg)

    # get the pullback of the absorption
    _, absorb_pb = pullback(absorb_mcf_gauge_transform, state, mcf_env)

    # initialize the partial pullbacks of the fixed point equations
    FP = generate_mcf_fixedpoint(state)

    # check if fixed-point equations are actually satisfied
    FPS = FP(state, mcf_env)
    fp_nrms = norm.(FPS)
    any(fp_nrms .> 1.0e1 * alg.tol)  &&
        @warn "Fixed-point equations not satisfied, still using the gradient: $fp_nrms"

    # start from the full automatic pullback
    _, fixedpoint_vjp = pullback(FP, state, mcf_env)

    # restrict to the pure environment pullback
    vjp_env(x) = fixedpoint_vjp(x)[2]

    # restrict to state pullback
    vjp_state(x) = fixedpoint_vjp(x)[1]

    function gaugefix_pullback(Δx_)
        Δself = NoTangent()
        Δalg = NoTangent()
        Δenv0 = ZeroTangent()
        Δsvd_alg = NoTangent()

        Δstate_gauged, Δweights, Δenv0 = unthunk.(Δx_)
        if Δenv0 isa AbstractZero
            Δenv0 = zerovector.(mcf_env)
        end

        # get the first part of the state adjoint from the pullback of the absorption
        Δstate0, Δenv1 = absorb_pb(Δstate_gauged)

        # get the second part of the state adjoint from the pullback of the gauge-fixing fixed-point condition
        # first accumulate gauge-fixing environment adjoints
        Δenv = add(Δenv0, Δenv1)
        # then solve linear problem to invert environment pullback
        Δa, info = reallinsolve(vjp_env, Δenv, Δenv, gradmode.solver_alg)
        if gradmode.solver_alg.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end
        # check if the linear problem actually converged
        linres = add(Δenv, vjp_env(Δa), -1)
        if norm(linres) > 1.0e2 * gradmode.solver_alg.tol
            msg = "gradient fixed-point iteration did not actually converge:"
            msg *= "\n  ‖ b - A x ‖ = $(norm(linres))"
            @warn msg
        end
        # finally plug it into the state pullback to get the contribution to the state cotangent
        Δstate1 = (-1) * vjp_state(Δa)

        # accumulate direct and indirect contributions to state cotangent
        Δstate = add(Δstate0, Δstate1)

        return Δself, Δstate, Δalg, Δenv0, Δsvd_alg
    end

    return (state_gauged, weights, mcf_env), gaugefix_pullback
end
