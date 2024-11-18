
"""
Contract the AbstractEnvironmentTensor with AbstractMPOTensor / AbstractEnvironmentTensor
Especially for 1-site projective H action
"""
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,2,-3] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[1,2,-4] * mpo.A[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,2,-4] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,3,1] * A.A[1,2,-3] * mpo.A[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3]
    return tmp
end
function contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3]
    return tmp
end
"""
Contract the AbstractEnvironmentTensor with AbstractMPOTensor / AbstractEnvironmentTensor
Especially for 2-site projective H action
"""
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,1] * A.A[1,2,-3,-4] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,1] * A.A[1,2,-3,-5] * mpo.A[-2,-4,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,3,1] * A.A[1,2,-3,-4] * mpo.A[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-4,1] * A.A[1,2,-3,-5] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,5}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,2,-4] * mpo.A[-3,2,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-2,1,-5] * mpo.A[-3,-4,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,-4] * mpo.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, Er::RightEnvironmentTensor{2})
    return El.A*Er.A
end
function contract(El::LeftCompositeEnvironmentTensor{3,5}, Er::RightEnvironmentTensor{3})
    return El.A*permute(Er.A,(2,1),(3,))
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * B.A[2,3,4,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,2,-2,3] * B.A[-1,3,4,1] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,-2,2,3] * C.A[4,5,6,-3] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 ;-2 -3] ≔ A.A[3,4,-3,5] * B.A[6,2,-2,3] * C.A[-1,5,6,1] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,1] * A.A[-1,1,2,-4] * EnvR.A[2,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,1] * A.A[-2,1,2,-6] * B.A[-1,2,3,-5] * EnvR.A[3,-4]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::DenseMPOTensor{4}, D::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
    @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A[1,2,-4]
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end


