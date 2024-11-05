
using TensorKit
include("../../src/iMPS.jl")

#= 

DMRG:
-U₁SU₂Fermion (Spinful Fermion)
-assigned state for fermion 


TDVP: spectrum function, structure factor
-spin evolution
-given operator evolution
-correaltion

=#


Lx = 8
Ly = 1

Latt = YCSqua(Lx,Ly)
Ndop = 1

ψ = let
    AuxSpace = vcat(Rep[U₁×SU₂]((Ndop, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], size(Latt) - 1))
    randMPS(U₁SU₂Fermion.PhySpace, AuxSpace)
end
Latt = YCSqua(Lx,Ly)

t = 1
U = 4

H = let 
    Root = InteractionTreeNode()
    LocalSpace = U₁SU₂Fermion

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.nd,i,"nd",U,nothing)
    end
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),t,LocalSpace.Z)
        addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end

D = 2^4

ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE)

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F⁺F,LocalSpace.FF⁺,LocalSpace.n), Latt, [k,0], (("Fₖ⁺","Fₖ"),("Fₖ","Fₖ⁺"),"n"),LocalSpace.Z)
    end

    calObs!(Obs,ψ)
end

Obs.values


