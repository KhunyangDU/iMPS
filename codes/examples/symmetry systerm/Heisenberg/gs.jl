
using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")


Lx = 7
Ly = 1

AuxSpace = repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly)
PhySpace = SU₂Spin.PhySpace 

ψ = randMPS(PhySpace,AuxSpace)

Latt = YCSqua(Lx,Ly)
J = 1
D = 2^4

H = Hamiltonian(Latt,J)

ψ, lsE = DMRG2!(ψ, H, D)
showQuantSweep(lsE)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = SU₂Spin

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SS,pair,("S","S"),nothing)
    end

    calObs!(Obs, ψ)
end

Obs.values

