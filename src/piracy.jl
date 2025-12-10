# type piracy patches

function ChainRulesCore.rrule(::typeof(unitcell), state::InfinitePEPS)
    tensors = unitcell(state)
    function unitcell_pullback(Δtensors_)
        Δtensors = unthunk(Δtensors_)
        return NoTangent(), InfinitePEPS(Δtensors)
    end
    return tensors, unitcell_pullback
end
