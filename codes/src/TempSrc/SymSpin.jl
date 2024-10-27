
module SU₂Spin

using TensorKit

const PhySpace = Rep[SU₂](1//2 => 1)

# S⋅S interaction
const SS = let
    AuxSpace = Rep[SU₂](1 => 1)
    OpL = TensorMap(ones, Float64, PhySpace, PhySpace ⊗ AuxSpace) * sqrt(3) / 2.
    OpR = permute(OpL', ((1,2), (3,)))
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

const S₊S₋ = let 
    AuxSpace = Rep[U₁](1 => 1)
    OpL = TensorMap(ones, PhySpace, PhySpace ⊗ AuxSpace)
    OpR = permute(OpL', ((1,2), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    map( x -> permute(x...), ( S₊S₋ |> y->((y[1]',((1,),(3,2))),(y[2]',((1,3),(2,)))) ) )
end

end

module TrivialSpin

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

end

