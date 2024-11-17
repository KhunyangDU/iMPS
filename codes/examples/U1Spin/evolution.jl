
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 16
Ly = 1
Latt = YCSqua(Lx,Ly)
@save "examples/U1Spin/data/Latt_$(Lx)x$(Ly).jld2" Latt

J = 1
H = Hamiltonian(Latt,J,J/2)

D = 2^6
ψ = let 
    AuxSpaces = vcat(Rep[U₁](mod(size(Latt),2) // 2 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt)//2:1//2:size(Latt)//2),], Lx*Ly-1))
    L = length(AuxSpaces)
    PhySpaces = repeat([U₁Spin.PhySpace,],L)
    push!(AuxSpaces, trivial(PhySpaces[1]))
    tmp = Vector{MPSTensor}(undef,L)
    for i in 1:L
        siteTensor = TensorMap(zeros,AuxSpaces[i] ⊗ AuxSpaces[i+1]', PhySpaces[i]')
        if isodd(i)
            block(siteTensor, Irrep[U₁](-1//2)) .= 1
        else
            block(siteTensor, Irrep[U₁](1//2)) .= 1
        end
        tmp[i] = MPSTensor(permute(siteTensor,(1,3),(2,)))
    end

    obj = MPS{L,Float64}(tmp)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)
    obj
end

T = 2
Nt = 40

lsψ, lst = TDVP2!(ψ, H, T, Nt, D)

Obs = let 
    Obs = MPSObservable()
    LocalSpace = U₁Spin

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
    end

    Obs
end

lsObs = Vector{Dict}(undef,length(lsψ))

for (ind,ψi) in enumerate(lsψ)
    @show ind
    @time "calculate observables" begin
        calObs!(Obs, ψi; destroy = false)
    end
    lsObs[ind] = Obs.values
end

@save "examples/U1Spin/data/lst_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.jld2" lst
@save "examples/U1Spin/data/lsψ_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.jld2" lsψ
@save "examples/U1Spin/data/lsObs_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.jld2" lsObs

