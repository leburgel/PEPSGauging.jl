# derivative of BP gauge fixing

# TODO

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
