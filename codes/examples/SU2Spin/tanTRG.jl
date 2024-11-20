using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/SU2Spin/data/Latt_$(Lx)x$(Ly).jld2" Latt

J = 1
D = 2^8

H = Hamiltonian(Latt,J)
ρ = let 
    ρ = IdDenseMPO(SU₂Spin.PhySpace, vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], L-1)))
    canonicalize!(ρ,1)
    normalize!(ρ)
    ρ
end

lsβ = vcat(2. .^ (-20:2:-1), 1:1)

SETTN!(lsβ[1], H, ρ; D=D)

lsρ = tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15)
@save "examples/SU2Spin/data/lsβ_$(Lx)x$(Ly)_$(D).jld2" lsβ
@save "examples/SU2Spin/data/lsρ_$(Lx)x$(Ly)_$(D).jld2" lsρ
