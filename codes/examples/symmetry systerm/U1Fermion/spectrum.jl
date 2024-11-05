using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

Ndop = 2
Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)

ψ = let 
    # design a state for symmetric fermion
end

#= t = 1

H = let 
    Root = InteractionTreeNode()
    LocalSpace = U₁Fermion
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),t,LocalSpace.Z)
        addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end

D = 2^4

ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE) =#

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁Fermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F⁺F,LocalSpace.FF⁺,LocalSpace.n), Latt, [k,0], (("Fₖ⁺","Fₖ"),("Fₖ","Fₖ⁺"),"n"),LocalSpace.Z)
    end

    calObs!(Obs,ψ)
end

@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values


