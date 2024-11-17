module U₁SU₂Fermion

using TensorKit

const PhySpace = Rep[U₁×SU₂]((-1,0) => 1, (0,1/2) => 1, (1,0) => 1)
    
const Z = let 
    tmp = TensorMap(ones,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](0,1/2)) .= -1
    tmp
end

const n = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
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
    F⁺ = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    block(F⁺, Irrep[U₁×SU₂](1, 0)) .= -sqrt(2)
    F = permute(F⁺', (2,1), (3,))
    block(F, Irrep[U₁×SU₂](1, 0)) .= sqrt(2)
    F⁺, F
end

const FF⁺ = let 
    AuxSpace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1; -2 -3] ≔ F⁺F[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F⁺F[2]'[-1,-3,1] * rev[1,-2]
    F, F⁺
end

const SS = let
    AuxSpace = Rep[U₁×SU₂]((0,1) => 1)
    OpL = TensorMap(ones, Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(3) / 2.
    OpR = permute(OpL', (2,1), (3,))
    OpL, OpR
end

end


module U₁U₁Fermion

using TensorKit

const PhySpace = Rep[U₁×U₁]((-1,0) => 1, (0,1 // 2) => 1, (0,-1 // 2) => 1,(1,0) => 1)
    
const Z = let 
    tmp = TensorMap(ones,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,1/2)) .= -1
    block(tmp,Irrep[U₁×U₁](0,-1/2)) .= -1
    tmp
end

const n₊ = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,1/2)) .= 1
    block(tmp,Irrep[U₁×U₁](1,0)) .= 1
    tmp
end

const n₋ = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,-1/2)) .= 1
    block(tmp,Irrep[U₁×U₁](1,0)) .= 1
    tmp
end

const n = n₊ + n₋

const nd = n₊ * n₋

const Sz = (n₊ - n₋) / 2

const F₊⁺F₊ = let
    AuxSpace = Rep[U₁×U₁]((1,1/2) => 1)
    F⁺ = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    F = TensorMap(ones, PhySpace ⊗ AuxSpace, PhySpace)
    F⁺, F
end

const F₋⁺F₋ = let
    AuxSpace = Rep[U₁×U₁]((1,-1/2) => 1)
    F⁺ = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    F = TensorMap(ones, PhySpace ⊗ AuxSpace, PhySpace)
    block(F⁺, Irrep[U₁×U₁](1, 0)) .= -1
    block(F, Irrep[U₁×U₁](1, 0)) .= -1
    F⁺, F
end

const F₊F₊⁺ = let 
    AuxSpace = Rep[U₁×U₁]((1, 1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1;-2 -3] ≔ F₊⁺F₊[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F₊⁺F₊[2]'[-1,-3,1] * rev[1,-2]
    -F, -F⁺
end

const F₋F₋⁺ = let 
    AuxSpace = Rep[U₁×U₁]((1, -1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1; -2 -3] ≔ F₋⁺F₋[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F₋⁺F₋[2]'[-1,-3,1] * rev[1,-2]
    -F, -F⁺
end

const S₊S₋ = let 
    AuxSpace = Rep[U₁×U₁]((0,1) => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁×U₁]((0,-1) => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

end

module TrivialSpinfulFermion
using TensorKit
function diagm(dg::Vector{T}) where T
    L = length(dg)
    mat = zeros(T,L,L)
    for (dgi,dge) in enumerate(dg)
        mat[dgi,dgi] = dge
    end
    return mat
end
const PhySpace = ℂ^4

function diagm(pair::Pair{Int64, Vector{T}}) where T
    L = length(pair[2]) + abs(pair[1])
    mat = zeros(T,L,L)
    if pair[1] > 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii,ii+pair[1]] = ie
        end
    elseif pair[1] < 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii-pair[1],ii] = ie
        end
    else
        mat = diagm(pair[2])
    end
    
    return mat
end

const Z = TensorMap(diagm([1,-1,-1,1]),PhySpace,PhySpace)
const F₊⁺ = TensorMap(diagm(2 => [1,1]),PhySpace,PhySpace)
const F₋⁺ = TensorMap(diagm(1 => [1,0,1]),PhySpace,PhySpace)
const nd = TensorMap(diagm([2,0,0,0]),PhySpace,PhySpace)
const F₊ = F₊⁺'
const F₋ = F₋⁺'

const F₊⁺F₊ = F₊⁺, F₊
const F₊F₊⁺ = F₊, F₊⁺

const F₋⁺F₋ = F₋⁺, F₋
const F₋F₋⁺ = F₋, F₋⁺

#const FF⁺ = F₊ + F₋, F₊⁺ + F₋⁺
#const F⁺F = F₊⁺ + F₋⁺, F₊ + F₋

const n₊ = F₊⁺*F₊
const n₋ = F₋⁺*F₋

const n = F₊⁺*F₊ + F₋⁺*F₋
const nn = n, n
end



module U₁Fermion 

using TensorKit

const PhySpace = Rep[U₁](-1//2 => 1, 1//2 => 1)

const Z = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁](1//2)) .= -1
    block(tmp,Irrep[U₁](-1//2)) .= 1
    tmp
end

const n = let 
    tmp = TensorMap(zeros,PhySpace,PhySpace)
    block(tmp,Irrep[U₁](1//2)) .= 1
    tmp
end

F⁺F = let 
    AuxSpace = Rep[U₁](1 => 1)
    F⁺ = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace )
    F = permute(F⁺', (2,1), (3,))
    F⁺, F
end

const FF⁺ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    F = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    F⁺ = permute(F', (2,1), (3,))
    F, F⁺
end

end

module TrivialSpinlessFermion
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


