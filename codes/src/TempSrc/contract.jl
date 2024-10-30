
"""
Contract the AbstractEnvironmentTensor with AbstractMPOTensor / AbstractEnvironmentTensor
Especially for 1-site projective H action
"""
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.t[-1,1] * A.A[1,2,-3] * mpo.t[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ El.t[-1,1] * A.A[1,2,-4] * mpo.t[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ El.t[-1,-3,1] * A.A[1,2,-4] * mpo.t[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.t[-1,3,1] * A.A[1,2,-3] * mpo.t[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.t[-1,-2,1] * Er.t[1,-3]
    return tmp
end
function contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.t[-1,-2,2,1] * Er.t[1,2,-3]
    return tmp
end
"""
Contract the AbstractEnvironmentTensor with AbstractMPOTensor / AbstractEnvironmentTensor
Especially for 2-site projective H action
"""
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.t[-1,1] * A.A[1,2,-3,-4] * mpo.t[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.t[-1,1] * A.A[1,2,-3,-5] * mpo.t[-2,-4,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.t[-1,3,1] * A.A[1,2,-3,-4] * mpo.t[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.t[-1,-4,1] * A.A[1,2,-3,-5] * mpo.t[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,5}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.t[-1,-2,1,2,-4] * mpo.t[-3,2,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.t[-1,-2,1,-5] * mpo.t[-3,-4,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.t[-1,-2,1,-4] * mpo.t[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, Er::RightEnvironmentTensor{2})
    return El.t*Er.t
end
function contract(El::LeftCompositeEnvironmentTensor{3,5}, Er::RightEnvironmentTensor{3})
    return El.t*permute(Er.t,(2,1),(3,))
end

