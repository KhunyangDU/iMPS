using TensorKit,JLD2,FiniteLattices
include("../model.jl")
include("../../../src/iMPS.jl")

Lx = 6
Ly = 6
Latt = YCSqua(Lx,Ly)
@save "examples/Free Fermion/data/$(Lx)x$(Ly)/Latt_$(Lx)x$(Ly).jld2" Latt

d = 2

lsμ = -4.0:0.2:4.0
@save "examples/Free Fermion/data/$(Lx)x$(Ly)/lsμ_$(Lx)x$(Ly).jld2" lsμ

t = 1

D_MPS = 2^3
maxd = FindMaxDist(neighbor(Latt))
D_MPO = d*(2*maxd + 2)

LanczosLevel = D_MPO*d
Nsweep = 3

for μ in lsμ
    @show μ
    H = HamMPO(Latt;μ=μ)
    ψ = RandMPS(Lx*Ly)
    ψ,lsE = sweepDMRG2(ψ,H,Nsweep,LanczosLevel,D_MPS)

    showQuantSweep(lsE;name="Eg sweep")

    @save "examples/Free Fermion/data/$(Lx)x$(Ly)/ψ_D=$(D_MPS)_$(Lx)x$(Ly)_t=$(t)_μ=$(μ).jld2" ψ
    @save "examples/Free Fermion/data/$(Lx)x$(Ly)/lsE_D=$(D_MPS)_$(Lx)x$(Ly)_t=$(t)_μ=$(μ).jld2" lsE
end



