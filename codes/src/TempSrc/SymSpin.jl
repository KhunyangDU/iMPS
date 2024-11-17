
module SU₂Spin

using TensorKit

const PhySpace = Rep[SU₂](1//2 => 1)

# S⋅S interaction
const SS = let
    AuxSpace = Rep[SU₂](1 => 1)
    OpL = TensorMap(ones, Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(3) / 2.
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

end

module U₁Spin

using TensorKit

const PhySpace = Rep[U₁](1//2 => 1, -1//2 => 1)

const Sz = let 
    Op = TensorMap(ones, PhySpace, PhySpace )
    block(Op, Irrep[U₁](1//2)) .= 1/2
    block(Op, Irrep[U₁](-1//2)) .= -1/2
    Op
end

const SzSz = Sz, Sz

const S₊S₋ = let 
    AuxSpace = Rep[U₁](1 => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

end

module TrivialSpinOneHalf

using TensorKit

const PhySpace = ℂ^2

const Sx = let 
    MatOp = [0 1;1 0] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end

const Sy = let 
    MatOp = [0 -1im;1im 0] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end

const Sz = let 
    MatOp = [1 0; 0 -1] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end

const S₊ = Sx + 1im * Sy
const S₋ = S₊'

const SxSx = Sx,Sx 
const SySy = Sy,Sy 
const SzSz = Sz,Sz 


end

