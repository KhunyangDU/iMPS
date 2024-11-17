using JLD2, CairoMakie, FiniteLattices


function dot(A,B)
    return sum(A .* B)
end

Lx = 16
Ly = 1
Latt = load("../codes/examples/U1Spin/data/Latt_$(Lx)x$(Ly).jld2")["Latt"]

D = 2^6
T = 2
Nt = 40

lst = load("../codes/examples/U1Spin/data/lst_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.jld2")["lst"]
lsObs = load("../codes/examples/U1Spin/data/lsObs_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.jld2")["lsObs"]

Sz = zeros(size(Latt),length(lst))

for i in eachindex(lst)
    Sz[:,i] = [lsObs[i]["Sz"][(j,)] for j in 1:size(Latt)]
end
    
width,height = 0.7 .* (600,200)

fig = Figure()
ax = Axis(fig[1,1],
xticks = 1:Lx,
xlabel = L"\text{site}\ i",
ylabel = L"tJ",
title = "$(Lx)x$(Ly) Heisenberg (D=$(D))",
titlealign = :left,
width = width,height = height)

hm = heatmap!(ax,1:size(Latt),lst,Sz,
colormap = :bwr,
#colorrange = (-0.3,0.3)
)
xlims!(ax,0.5,size(Latt) + 0.5)
#ylims!(ax,0,8)

Colorbar(fig[1,2],hm,
label = L"\langle S^z \rangle")


resize_to_layout!(fig)
display(fig)

save("U1Spin/figures/quench__$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_AFM.png",fig)

