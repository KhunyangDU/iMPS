using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

Latt = YCSqua(Lx,Ly)
@save "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
H = Hamiltonian(Latt;params...)
D = 2^8

ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly)
    ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    normalize!(ρ)
    ρ
end

lsβ = vcat(2. .^ (-20:2:-1), 1:1)

SETTN!(lsβ[1], H, ρ; D=D)
lsρ = tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15)

@save "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
