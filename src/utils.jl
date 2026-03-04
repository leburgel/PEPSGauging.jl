# convenience tools

const BondTensor{S} = AbstractTensorMap{T, S, 1, 1} where {T}
const SquareTensorMap{S, N} = AbstractTensorMap{T, S, N, N} where {T}

const BondTensors{TB} = Matrix{TB} where {TB <: BondTensor}

function project_hermitian(x::BondTensor)
    return (x + x') / 2
end

function project_traceless(x::BondTensor)
    n = dim(domain(x))
    return x - tr(x) / n * id(storagetype(x), domain(x))
end

project_traceless_hermitian = project_traceless ∘ project_hermitian
