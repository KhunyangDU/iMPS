module U₁SU₂Fermion

using TensorKit

const PhySpace = Rep[U₁×SU₂]((-1,0) => 1, (0,1/2) => 1, (1,0) => 1)
    
const Z = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](-1,0)) .= 1
    block(tmp,Irrep[U₁×SU₂](0,1/2)) .= -1
    block(tmp,Irrep[U₁×SU₂](1,0)) .= 1
    tmp
end

const n = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](-1,0)) .= 0
    block(tmp,Irrep[U₁×SU₂](0,1/2)) .= 1
    block(tmp,Irrep[U₁×SU₂](1,0)) .= 2
    tmp
end

const nd = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](1,0)) .= 1
    tmp
end

const F⁺F = let 
    AuxSpace = Rep[U₁×SU₂]((1,1/2) => 1)
    F⁺ = TensorMap(ones, PhySpace, PhySpace ⊗ AuxSpace)
    F = permute(F⁺', (1,2), (3,))
    F⁺, F
end

const FF⁺ = let 
    AuxSpace = Rep[U₁×SU₂]((-1,1/2) => 1)
    F = TensorMap(ones, PhySpace, PhySpace ⊗ AuxSpace)
    F⁺ = permute(F', (1,2), (3,))
    F, F⁺
end

const SS = let
    AuxSpace = Rep[U₁×SU₂]((0,1) => 1)
    OpL = TensorMap(ones, Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(3) / 2.
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

end

module U₁Fermion 

using TensorKit

const PhySpace = Rep[U₁](0 => 1, 1 => 1, -1 => 1)

const Z = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁](-1)) .= 1
    block(tmp,Irrep[U₁](0)) .= -1
    block(tmp,Irrep[U₁](1)) .= 1
    tmp
end

const n = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁](-1)) .= 0
    block(tmp,Irrep[U₁](0)) .= 1
    block(tmp,Irrep[U₁](1)) .= 2
    tmp
end

const nd = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁](1)) .= 1
    tmp
end

F⁺F = let 
    AuxSpace = Rep[U₁](1 => 1)
    F⁺ = TensorMap(ones, PhySpace, PhySpace ⊗ AuxSpace)
    F = permute(F⁺', (1,2), (3,))
    F⁺, F
end

const FF⁺ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    F = TensorMap(ones, PhySpace, PhySpace ⊗ AuxSpace)
    F⁺ = permute(F', (1,2), (3,))
    F, F⁺
end

end

module TrivialFermion
using TensorKit
const PhySpace = ℂ^2
const Z = TensorMap([-1 0; 0 1],PhySpace,PhySpace)
const F = TensorMap([0 0;1 0],PhySpace,PhySpace)
const F⁺ = TensorMap([0 1;0 0],PhySpace,PhySpace)
const FF⁺ = F, F⁺
const F⁺F = F⁺, F
const n = F⁺*F
const nn = n, n
end

