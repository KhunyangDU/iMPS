using JLD2, CairoMakie, FiniteLattices


function dot(A,B)
    return sum(A .* B)
end

Lx = 4
Ly = 4
Latt = load("../codes/examples/SU2Spin/data/Latt_$(Lx)x$(Ly).jld2")["Latt"]

J = 1
D = 2^10
T = 2/J
Nt = 20

lst = load("../codes/examples/SU2Spin/data/lst_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt).jld2")["lst"]
lsObs = load("../codes/examples/SU2Spin/data/lsObs_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt).jld2")["lsObs"]


maxkrx = pi
maxkry = pi
Nkx = 25
Nky = 25
kx = range(-maxkrx,maxkrx,Nkx)
ky = range(-maxkry,maxkry,Nky)

figsize = (width = 300,height = 300)

for (ind,Obs) in enumerate(lsObs[1:10])
    SS = Dict()
    for i in 1:size(Latt),j in i:size(Latt)
        pair = (i,j)
        if i==j 
            SS[pair] = 3/4
        else
            SS[pair] = Obs["SS"][pair]
        end
    end

    fac = zeros(Nkx,Nky)
    for (ix,kvx) in enumerate(kx),(iy,kvy) in enumerate(ky)
        kv = kvx,kvy
        fac[ix,iy] = let 
            tmp_od = 0
            tmp_d = 0
            for pair in keys(SS)
                if pair[1] == pair[2]
                    tmp_d += exp(1im*dot(kv,coordinate(Latt,pair[1]) .- coordinate(Latt,pair[2]))) * SS[pair] / size(Latt)
                else
                    tmp_od += exp(1im*dot(kv,coordinate(Latt,pair[1]) .- coordinate(Latt,pair[2]))) * SS[pair] / size(Latt)
                end
            end
            2*real(tmp_od) + tmp_d
        end 
    end

    fig = Figure()
    ax = Axis(fig[1,1],autolimitaspect  = 1,title="t=$(round(lst[ind];digits=3))";figsize...)
    hm = heatmap!(ax,kx / pi, ky / pi, fac,colorrange = (0,4))
    #Colorbar(fig[1,2],hm)

    resize_to_layout!(fig)
    display(fig)
    
end





