using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 4
Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt)//2 :1//2:size(Latt)//2 ),], Lx*Ly-1))
    randMPS(U₁Spin.PhySpace ,AuxSpace)
end


J = 1
D = 2^10

H = Hamiltonian(Latt,J,J/2)

ψ, lsE = DMRG2!(ψ, H, D;Nsweep=5,LanczosLevel = 20)
showQuantSweep(lsE)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁Spin

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    end

    calObs!(Obs, ψ)
end
Obs.values




