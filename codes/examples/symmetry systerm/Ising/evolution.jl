
using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

Lx = 7
Ly = 1


ψ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly)
    PhySpace = TrivialSpinOneHalf.PhySpace 
    L = length(AuxSpaces)
    push!(AuxSpaces, trivial(PhySpace))
    tmp = Vector{MPSTensor}(undef,L)
    for i in 1:L
        tmp[i] = MPSTensor([1.,0.], AuxSpaces[i] ⊗ PhySpace, AuxSpaces[i+1])
    end

    obj = MPS{L,Float64}(tmp)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)
    obj
end


Latt = YCSqua(Lx,Ly)
J = -1
h = 0
D = 2^4

#= H = Hamiltonian(Latt,J,h,0.1)
ψ, lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE) =#

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