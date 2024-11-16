using JLD2, CairoMakie, BenchmarkTools

Lx = 8
Ly = 4

lsD = load("../codes/examples/benchmark/data/lsD_$(Lx)x$(Ly).jld2")["lsD"]
lsinfo = load("../codes/examples/benchmark/data/lsinfo_$(Lx)x$(Ly).jld2")["lsinfo"]

inner = length(findall(x -> x == lsD[1],lsD))
lsD
lsMemory = map(x -> minimum(x).memory, lsinfo) ./ 2^30 # GiB
lsTime = map(x -> minimum(x).time, lsinfo) ./ 10^9 # s

lsD, lsTime, lsMemory = map(x -> x[2:inner:end], (lsD, lsTime, lsMemory))


lsTime, lsMemory

using CairoMakie

# 创建数据

# 初始化Figure
fig = Figure()
figsize = (width = 300,height = 200)

ax1 = Axis(fig[1, 1], title = "Benchmark ($(Lx)x$(Ly) square Heisenberg)", xlabel = "Bond Dimension", ylabel = "Sweep Time / s",
ytickcolor = :blue, yticklabelcolor = :blue, ylabelcolor=:blue,
xminorticksvisible = true, xminorgridvisible = true,
xminorticks = IntervalsBetween(5),
xscale = log10,
leftspinecolor=:blue,rightspinecolor = :red;figsize...)
scatter!(ax1, lsD, lsTime, color = :blue)

ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "Used Memory / GiB",
xscale = log10,yscale = log10,
ytickcolor = :red, yticklabelcolor = :red, ylabelcolor=:red)
scatter!(ax2, lsD, lsMemory, color = :red)

# 配置图例
ylims!(ax1,1,100)
ylims!(ax2,1,100)
hidespines!(ax2)
hidexdecorations!(ax2)
resize_to_layout!(fig)
# 显示图表
display(fig)

save("benchmark/figures/bond dimension_$(Lx)x$(Ly).pdf",fig)

