using TensorKit
include("../../src/iMPS.jl")
include("model.jl")


Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^8

@load "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@show 1 ./ lsβ
f = zeros(length(lsβ))
u = zeros(length(lsβ))

for (i,ρ) in enumerate(lsρ)
    f[i] = -log(tr(ρ)) / lsβ[i] / size(Latt)
end
f
