using TensorKit, CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

function easyinterp10(v,N=100)
    return 10. .^ (range(log10.(extrema(v))..., N))
end

#= 
0126：
code
注意在不同type之间控制D和trunc err
每次计算输出纠缠熵
learn
ED for Hubbard
bethe ansatz

long range target
有限温度对称性
资源监视系统
包管理与编译
=#

#= 
解决数据对不上
写有限温度的calObs
=#


Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^8

H = Hamiltonian(Latt;params...)

@load "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
f = zeros(length(lsβ))
u = zeros(length(lsβ))

for (i,ρ) in enumerate(lsρ)
    Z = tr(ρ)
    f[i] = -log(Z) / lsβ[i] / size(Latt) / 2
    u[i] = tr(ρ, H) / Z
end

lsT1 = centralize(1 ./ lsβ)
lsβ1 = centralize(lsβ)
Ce = - centralize(lsβ) .* diff(u) ./ diff(log.(lsβ))

cβ = easyinterp10(lsβ)

figsize = (height=150,width=300)
fig = Figure()
axf = Axis(fig[1,1];xscale=log10,figsize...)
#ylims!(axf,-10,1)
scatter!(axf, 1 ./ lsβ, f)
lines!(axf, 1 ./ cβ, fe.(cβ,L);color = :red)

axu = Axis(fig[2,1];xscale=log10,figsize...)
scatter!(axu, 1 ./ lsβ, u)
#lines!(axu, 1 ./ lsβ, fe.(lsβ,L);color = :red)

axce = Axis(fig[3,1];xscale=log10,figsize...)
scatter!(axce, lsT1, Ce)
lines!(axce, 1 ./ cβ, ce.(cβ,L);color = :red)

resize_to_layout!(fig)
display(fig)

save("examples/TrivialSpinlessFermion/figures/thermal quantity.png",fig)


