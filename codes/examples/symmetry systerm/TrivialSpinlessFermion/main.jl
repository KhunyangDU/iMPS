using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

μ = 0
t = -1
H = Hamiltonian(Latt, t,μ)
D = 2^4

ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE)

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinlessFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F⁺F,LocalSpace.FF⁺,LocalSpace.n), Latt, [k,0], (("Fₖ⁺","Fₖ"),("Fₖ","Fₖ⁺"),"n"),LocalSpace.Z)
    end

    calObs!(Obs,ψ)
end

Obs.values
