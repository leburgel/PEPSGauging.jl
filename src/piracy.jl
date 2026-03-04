# type piracy patches

function ChainRulesCore.rrule(::typeof(unitcell), state::InfinitePEPS)
    tensors = unitcell(state)
    function unitcell_pullback(Δtensors_)
        Δtensors = unthunk(Δtensors_)
        return NoTangent(), InfinitePEPS(Δtensors)
    end
    return tensors, unitcell_pullback
end

function ChainRulesCore.rrule(::Type{InfinitePEPS}, A::Matrix{<:PEPSTensor})
    network = InfinitePEPS(A)
    function InfinitePEPS_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        return NoTangent(), unitcell(Δnetwork)
    end
    return network, InfinitePEPS_pullback
end

# if we need this, something went wrong...
_reconstruct_tangent(Δy, _) = Δy
function _reconstruct_tangent(Δy::Tangent, x)
    return ChainRulesCore.construct(typeof(x), ChainRulesCore.backing(Δy))
end
