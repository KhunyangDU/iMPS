
using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

Lx = 7
Ly = 1

AuxSpace = repeat([ℂ^1,], Lx*Ly)
PhySpace = TrivialSpinOneHalf.PhySpace 

ψ = randMPS(PhySpace,AuxSpace)

Latt = YCSqua(Lx,Ly)
J = -1
h = 0
D = 2^4

H = Hamiltonian(Latt,J,h,1)

ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf
    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    end

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    end

    calObs!(Obs, ψ)
end

Obs.values

