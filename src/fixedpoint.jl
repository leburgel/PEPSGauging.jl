#
# BP gauging fixed-point equations
#

# TODO: actually sort this out

#
# MCF gauging fixed-point equations
#

# just Fig. 8 of https://arxiv.org/abs/2209.14358 for the single-site unit cell case
# for the general case, I just guessed the stupidest modification, and it seems to work?
function generate_mcf_fixedpoint(statefp::InfinitePEPS)
    function mcf_fixedpoint(state::InfinitePEPS, env::MCFEnv)
        # project input
        env = _project_input(env)

        # absorb gauge transform
        state´ = absorb_mcf_gauge_transform(state, env)

        # extract fixed-point conditions from gauge-fixed tensors
        FP12 = map(eachcoordinate(state´)) do (r, c)
            # get the gauged tensor, and the one above and to the right
            A´ = state´[r, c]
            Ab´ = state´[_prev(r, end), c]
            Ar´ = state´[r, _next(c, end)]

            # get reduced density matrices
            @tensor ρN[-1; -2] := A´[1; -2 2 3 4] * conj(A´[1; -1 2 3 4])
            @tensor ρS[-1; -2] := Ab´[1; 2 3 -1 4] * conj(Ab´[1; 2 3 -2 4])

            @tensor ρE[-1; -2] := A´[1; 2 -2 3 4] * conj(A´[1; 2 -1 3 4])
            @tensor ρW[-1; -2] := Ar´[1; 2 3 4 -1] * conj(Ar´[1; 2 3 4 -2])

            # the reduced density matrix conditions
            fp1 = ρN - ρS
            fp2 = ρE - ρW

            return fp1, fp2
        end
        FP1 = map(fps -> fps[1], FP12)
        FP2 = map(fps -> fps[2], FP12)
        return FP1, FP2
    end
    return mcf_fixedpoint
end
