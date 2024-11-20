_getD(A::DenseMPOTensor{4}) = dims(domain(A.A))[1]
_getD(A::LeftEnvironmentTensor) = dims(domain(A.A))[end]
_getD(A::RightEnvironmentTensor) = dims(codomain(A.A))[end]
_getD(A::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor}) = _getD(A.A[1])
_getD(A::Union{CompositeMPSTensor,CompositeMPOTensor}) = dims(domain(A.A))[1]

function Main.dims(A::Union{LeftEnvironmentTensor,RightEnvironmentTensor,DenseMPOTensor,LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor})
    return dims(codomain(A.A)),dims(domain(A.A))
end