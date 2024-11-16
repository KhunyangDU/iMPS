
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 4

AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
PhySpace = SU₂Spin.PhySpace 

ψ = randMPS(PhySpace,AuxSpace)

Latt = YCSqua(Lx,Ly)
J = 1
D = 2^10

H = Hamiltonian(Latt,J)

ψ, lsE = DMRG2!(ψ, H, D;Nsweep=5,LanczosLevel = 25)
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

