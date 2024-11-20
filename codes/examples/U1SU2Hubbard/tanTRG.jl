using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/U1SU2Hubbard/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2^10
Ndop = 0
params = (U = 0,μ=0)
H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = vcat(Rep[U₁×SU₂]((Ndop, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], size(Latt) - 1))
    ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    normalize!(ρ)
    ρ
end

lsβ = vcat(2. .^ (-15:1:-1), 1:10)

SETTN!(lsβ[1], H, ρ;D=D)
tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15,TruncErr=1e-2)

@save "examples/U1SU2Hubbard/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ


