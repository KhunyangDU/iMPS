using JLD2, CairoMakie, BenchmarkTools,LaTeXStrings, LsqFit

Lx = 8
Ly = 4

lsD = load("../codes/examples/benchmark/data/lsD_$(Lx)x$(Ly).jld2")["lsD"]


#lsinfo = load("../codes/examples/benchmark/data/lsinfo_$(Lx)x$(Ly).jld2")["lsinfo"]
N = 14
lsinfo = Vector(undef,N)
selectedD = Vector(undef,N)
for i in 1:N
    lsinfo[i] = load("../codes/examples/benchmark/data/tmpinfo_$(i+4)_$(Lx)x$(Ly).jld2")["tmpinfo"]
    selectedD[i] = lsD[i+4]
end


inner = length(findall(x -> x == lsD[1],lsD))
lsMemory = map(x -> minimum(x).memory, lsinfo) ./ 2^30 # GiB
lsTime = map(x -> minimum(x).time, lsinfo) ./ 10^9 # s
selectedD, lsTime, lsMemory = map(x -> x[2:inner:end], (selectedD, lsTime, lsMemory))

fig = Figure()
figsize = (width = 400,height = 200)

ax1 = Axis(fig[1, 1], title = "Benchmark ($(Lx)x$(Ly) square SU₂ Heisenberg)", xlabel = "Bond Dimension", ylabel = "Sweep Time / s",
ytickcolor = :blue, yticklabelcolor = :blue, ylabelcolor=:blue,
xminorticksvisible = true, xminorgridvisible = true,
yminorticksvisible = true, yminorgridvisible = true,
xscale = log10,yscale = log10,
xticks = (500:500:5000 |> x -> (x,map(string,x))),
leftspinecolor=:blue,rightspinecolor = :red;figsize...)
scatter!(ax1, selectedD, lsTime, color = :blue)

ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "Used Memory / GiB",
xscale = log10,yscale = log10,
ytickcolor = :red, yticklabelcolor = :red, ylabelcolor=:red)
scatter!(ax2, selectedD, lsMemory, color = :red)

@. model(x, p) = p[1]*x^p[2]
p0 = [0.5, 0.5]

fit1 = curve_fit(model, selectedD, lsTime, p0)
lines!(ax1,(range(extrema(selectedD)...,100) |> x-> (x,model(x,fit1.param)))...,
color = :blue,linestyle=:dash)

fit2 = curve_fit(model, selectedD, lsMemory, p0)
lines!(ax2,(range(extrema(selectedD)...,100) |> x-> (x,model(x,fit2.param)))...,
color = :red,linestyle=:dash)

text!(2250, 10^2.7, text = "Memory ∼ D^$(round(fit1.param[2];digits=3))", align = (:right,:center),color = :red)
text!(2750, 10^2.2, text = "Time ∼ D^$(round(fit2.param[2];digits=3))", align = (:left,:center),color = :blue)

ylims!(ax1,5e1,1.5e3)
ylims!(ax2,5e1,1.5e3)
xlims!(ax1,(extrema(selectedD).*(0.95,1.05))...)
xlims!(ax2,(extrema(selectedD).*(0.95,1.05))...)

hidespines!(ax2)
hidexdecorations!(ax2)
resize_to_layout!(fig)

display(fig)

save("benchmark/figures/bond dimension_$(Lx)x$(Ly).pdf",fig)

