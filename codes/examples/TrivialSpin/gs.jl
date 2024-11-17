using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

hx = 0
H = Hamiltonian(Latt;hx=hx)
D = 2^6

ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE)

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
    end

    calObs!(Obs,ψ)
end

Obs.values
