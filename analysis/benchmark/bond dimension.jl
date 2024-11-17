using JLD2, CairoMakie, BenchmarkTools

Lx = 8
Ly = 4

lsD = load("../codes/examples/benchmark/data/lsD_$(Lx)x$(Ly).jld2")["lsD"]


#lsinfo = load("../codes/examples/benchmark/data/lsinfo_$(Lx)x$(Ly).jld2")["lsinfo"]
N = 12
lsinfo = Vector(undef,N)
for i in 1:N
    lsinfo[i] = load("../codes/examples/benchmark/data/tmpinfo_$(i)_$(Lx)x$(Ly).jld2")["tmpinfo"]
end


inner = length(findall(x -> x == lsD[1],lsD))
lsD
lsMemory = map(x -> minimum(x).memory, lsinfo) ./ 2^30 # GiB
lsTime = map(x -> minimum(x).time, lsinfo) ./ 10^9 # s

lsD, lsTime, lsMemory = map(x -> x[2:inner:end], (lsD, lsTime, lsMemory))

lsTime, lsMemory

fig = Figure()
figsize = (width = 400,height = 200)

ax1 = Axis(fig[1, 1], title = "Benchmark ($(Lx)x$(Ly) square Heisenberg)", xlabel = "Bond Dimension", ylabel = "Sweep Time / s",
ytickcolor = :blue, yticklabelcolor = :blue, ylabelcolor=:blue,
xminorticksvisible = true, xminorgridvisible = true,
yminorticksvisible = true, yminorgridvisible = true,
xscale = log10,yscale = log10,
leftspinecolor=:blue,rightspinecolor = :red;figsize...)
scatter!(ax1, lsD[1:div(N,2)], lsTime, color = :blue)

ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "Used Memory / GiB",
xscale = log10,yscale = log10,
ytickcolor = :red, yticklabelcolor = :red, ylabelcolor=:red)
scatter!(ax2, lsD[1:div(N,2)], lsMemory, color = :red)

ylims!(ax1,1.5e1,1.5e3)
ylims!(ax2,1.5e1,1.5e3)
hidespines!(ax2)
hidexdecorations!(ax2)
resize_to_layout!(fig)

display(fig)

save("benchmark/figures/bond dimension_$(Lx)x$(Ly).pdf",fig)

