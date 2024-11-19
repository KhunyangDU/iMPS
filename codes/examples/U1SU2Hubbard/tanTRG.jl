using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
J = 1
D = 2^4
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

lsβ = vcat((3/2) .^ (-10:1:-1), 1:1/3:10)

SETTN!(lsβ[1], H, ρ)

tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15)
