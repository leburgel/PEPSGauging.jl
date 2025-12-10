# convenience tools

const BondTensor{S} = AbstractTensorMap{T, S, 1, 1} where {T}
const SquareTensorMap{S, N} = AbstractTensorMap{T, S, N, N} where {T}

const BondTensors{TB} = Matrix{TB} where {TB <: BondTensor}

function project_hermitian(x::BondTensor)
    return (x + x') / 2
end
