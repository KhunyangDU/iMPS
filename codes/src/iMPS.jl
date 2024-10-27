using TensorKit,JLD2,MKL,FiniteLattices


include("Environment/Environment.jl")
include("Environment/Initialize.jl")
include("Environment/Pushleft.jl")
include("Environment/Pushright.jl")

include("Operations/Move.jl")
include("Operations/Merge.jl")
include("Operations/Contract.jl")
include("Operations/SVD.jl")
include("Operations/Variation.jl")

include("MPS/MPS.jl")
include("MPS/Operations.jl")

include("MPO/MPO.jl")
include("MPO/Normalize.jl")
include("MPO/Operations.jl")
include("MPO/Operators.jl")
include("MPO/ObsMPO.jl")

include("Tools/Tools.jl")
include("Tools/geometry.jl")
include("Tools/algebra.jl")

include("IntrTree/LocalOperator.jl")
include("IntrTree/Node.jl")
include("IntrTree/addIntr.jl")
include("IntrTree/addIntr1.jl")
include("IntrTree/addIntr2.jl")
include("IntrTree/Automata.jl")

include("Observables/ObsTree.jl")
include("Observables/calObs.jl")
include("Observables/addObs.jl")
include("Observables/Observables.jl")

include("LocalSpace/Fermion.jl")
include("LocalSpace/Spin.jl")

include("Algorithm/DMRG.jl")
include("Algorithm/Lanczos.jl")
include("Algorithm/TDVP.jl")
include("Algorithm/SETTN.jl")
include("Algorithm/tanTRG.jl")



include("TempSrc/AbstractType.jl")
include("TempSrc/AbstractMPS.jl")
include("TempSrc/AbstractMPO.jl")
include("TempSrc/SymSpin.jl")
include("TempSrc/canonicalize.jl")


include("TempSrc/AbstractEnvironment.jl")
include("TempSrc/push.jl")


#= 
MPO data matrix should be the hermitian conjugate of the 
matrix of observables, i.e.
M_{data} = M_{Obs}⁺
under this convention, the evolution operator is exp(i)
=#

#= 
Suppose you have been familiar with TensorKit.jl
Indexing formalism of TensorMap
[] -> codomain, () -> domain

I. MPS
    1. Right Orthogonal:
           ___
    (2) → |   | → [1]
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1] , (2) ⊗ (3) }
           [2] 
            ↑
           ___
    [1] ← |   | ← (3)
           ‾‾‾

    i.e. TensorMap{ [1] ⊗ [2], (3)  }
    -------------------
    2. Left Orthogonal:
           ___
    [1] ← |   | ← (3)
           ‾‾‾
            ↑
           (2)
    i.e. TensorMap{ [1] , (2) ⊗ (3) }
           [1]
            ↑
           ___
    (3) → |   | → [2]
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2]  , (3) }
    -------------------
    3. Center:
           ___
    (1) → |   | ← (3)
           ‾‾‾
            ↑
           (2)
    i.e. TensorMap{ (1) ⊗ (2) ⊗ (3)}
           [2]
            ↑
           ___
    [1] ← |   | → [3]
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3]}
    -------------------
    4. SVD conjunction
    -Right
           ___
    (2) → | → | ← (1)
           ‾‾‾
    i.e. TensorMap{ (1) ⊗ (2) }

    -Left
           ___
    (1) → | ← | ← (2)
           ‾‾‾
    i.e. TensorMap{ (1) ⊗ (2) }
    -------------------
    5. 2-site Center MPS
           _______
    (1) → |       | ← (4)
           ‾‾‾‾‾‾‾
            ↑   ↑
           (2) (3)
    i.e. TensorMap{ (1) ⊗ (2) ⊗ (3) ⊗ (4) }
           [2] [3]
            ↑   ↑
           _______
    [1] ← |       | → [4]
           ‾‾‾‾‾‾‾
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3] ⊗ [4] }

II.MPO
    1.1 Local MPO (Right Orthogonal)
           [2]
            ↑
           ___
    (3) → |   | → [1]
           ‾‾‾
            ↑
           (4)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

           (4)
            ↓
           ___
    [1] ← |   | ← (3)
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    1.2 Local MPO (Left Orthogonal)
           [1]
            ↑
           ___
    [2] ← |   | ← (4)
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

           (3)
            ↓
           ___
    (4) → |   | → [2]
           ‾‾‾
            ↓
           [1]
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    1.3 Local MPO (Center Orthogonal)
           [1]
            ↑
           ___
    (2) → |   | ← (4)
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1], (2) ⊗ (3) ⊗ (4) }

           (4)
            ↓
           ___
    [1] ← |   | → [3]
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) }

    1.4 Local 2 site MPO (Center Orthogonal)
           [2] [1]
            ↑   ↑
           _______
    (3) → |       | ← (6)
           ‾‾‾‾‾‾‾
            ↑   ↑
           (4) (5)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) ⊗ (5) ⊗ (6) }

           (4)
            ↓
           ___
    [1] ← |   | → [3]
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) }

    2. 0-site Effective Hamiltonian
         [1] [2]
          ↑   ↑
         _______
        |       |
         ‾‾‾‾‾‾‾
          ↑   ↑
         (3) (4)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    3. 1-site Effective Hamiltonian
         [1] [2] [3]
          ↑   ↑   ↑
         ___________
        |           |
         ‾‾‾‾‾‾‾‾‾‾‾
          ↑   ↑   ↑
         (4) (5) (6)
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) ⊗ (5) ⊗ (6) }

    4. 2-site Effective Hamiltonian
         [1] [2] [3] [4]
          ↑   ↑   ↑   ↑
         _______________
        |               |
         ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
          ↑   ↑   ↑   ↑
         (5) (6) (7) (8)
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3] ⊗ [4], (5) ⊗ (6) ⊗ (7) ⊗ (8) }

III.Environment
    1. Right Environment
           ___
    [1] ← |   |
    (2) → |   |
    (3) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] , (2) ⊗ (3) }

    *canonicalized
           ___
    [1] ← |   |
    [2] ← |   |
    (3) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2], (3) }

           ___
    [1] ← |   |
    (2) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] , (2) }

    2. Left Environment
           ___
          |   | → [1]
          |   | → [2]
          |   | ← (3)
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2], (3)  }
           ___
          |   | → [1]
          |   | ← (2)
           ‾‾‾
    i.e. TensorMap{ [1] , (2)  }

 =#



