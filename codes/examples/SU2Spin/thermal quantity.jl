using TensorKit
include("../../src/iMPS.jl")
include("model.jl")


Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/SU2Spin/data/Latt_$(Lx)x$(Ly).jld2" Latt

J = 1
D = 2^8

@load "examples/SU2Spin/data/lsβ_$(Lx)x$(Ly)_$(D).jld2" lsβ
@load "examples/SU2Spin/data/lsρ_$(Lx)x$(Ly)_$(D).jld2" lsρ
@show 1 ./ lsβ
f = zeros(length(lsβ))
u = zeros(length(lsβ))

for (i,ρ) in enumerate(lsρ)
    f[i] = -log(tr(ρ)) / lsβ[i] / size(Latt) /2
end
f
