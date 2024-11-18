# Heisenberg
using TensorKit, BenchmarkTools, JLD2
include("../../src/iMPS.jl")

Lx = 8
Ly = 4
Latt = YCSqua(Lx,Ly)
@save "examples/benchmark/data/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

J = 1

H = let 
    Root = InteractionTreeNode()
    LocalSpace = SU₂Spin

    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SS,pair,("S","S"),J,nothing)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end


lsD = let
    lsD = broadcast(Int64 ∘ round, vcat(0.5:0.5:8) .* 1000)
    repeat(lsD, inner=2)
end
@save "examples/benchmark/data/lsD_$(Lx)x$(Ly).jld2" lsD

@time "Initialize Environment" begin
    Env = Environment([ψ,H,adjoint(ψ)])
    initialize!(Env)
end

lsinfo = Vector(undef,length(lsD))
for (i,D) in enumerate(lsD)
    i ≤ 16 && continue
    @show i
    tmpinfo = @benchmark DMRG2!($Env, $D;Nsweep = 4)
    @save "examples/benchmark/data/tmpinfo_$(i)_$(Lx)x$(Ly).jld2" tmpinfo
    lsinfo[i] = tmpinfo
end

@save "examples/benchmark/data/lsinfo_$(Lx)x$(Ly).jld2" lsinfo

