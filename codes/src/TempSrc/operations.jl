
"""
mul!(C, A, B, α, β) -> C
Combined inplace matrix-matrix or matrix-vector multiply-add \$A B α + C β\$.
The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of C, A, B.
Nsweep: times of variational calculation (sweep). Default is 2.
"""
function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}, α::Number, β::Number; kwargs...)
    #D_MPO = get(kwargs, :D_MPO, maximum(vcat(map(size, filter(x -> typeof(x) <: DenseMPO.[A,B])[1].ts)...)))
    D_MPO = get(kwargs, :D_MPO, maximum(vcat(collect.(map(size, filter(x -> typeof(x) <: DenseMPO, [A,B])[1].ts))...)))
    Nsweep = get(kwargs, :D_MPO, 2)

    tmp = RandDenseMPO(L)'
    @time "Initialize A,B,C Environment" begin
        EnvAB = Environment([deepcopy(A),deepcopy(B),tmp])
        EnvC = Environment([deepcopy(C),tmp])
        initialize!(EnvAB)
        initialize!(EnvC)
    end

    for i in 1:Nsweep
        totaltruncerror = 0
        @time "sweep $i finished, max truncation error = $(totaltruncerror)" begin
            println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                @show site
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α, β, map(z -> contract(z.envs[site].A, vcat(map(u -> z.layer[u].ts[site:site+1],1:length(z.layer)-1)...)..., z.envs[site+2].A),[EnvAB,EnvC])...)
                end; direction=:right,trunc = truncdim(D_MPO))
                map(z -> pushright!(z,map(DenseMPOTensor, [tl, tr])...),[EnvAB,EnvC])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
            println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α, β,map(z -> contract(z.envs[site-1].A, vcat(map(u -> z.layer[u].ts[site-1:site],1:length(z.layer)-1)...)..., z.envs[site+1].A),[EnvAB,EnvC])...)
                end; direction=:left,trunc = truncdim(D_MPO))
                map(z -> pushleft!(z,map(DenseMPOTensor, [tl, tr])...),[EnvAB,EnvC])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
        end
    end

    @assert EnvAB.layer[end] == EnvC.layer[end]
    return axpy!(EnvAB.layer[end]',C)
end
"""
axpy!(α, x, y) -> y
Overwrite y with x * α + y and return y. If x and y have the same axes, it's equivalent with y .+= x .* a.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of x, y.
Nsweep: times of variational calculation (sweep). Default is 2.

"""
function axpy!(α::Number, x::DenseMPO, y::DenseMPO;kwargs...)
    D_MPO = get(kwargs, :D_MPO, max(map(y -> maximum(vcat(collect.(map(size, y.ts))...)),[x,y])...))
    Nsweep = get(kwargs, :D_MPO, 3)

    tmp = RandDenseMPO(L)'
    @time "Initialize x,y Environment" begin
        Envx = Environment([deepcopy(x),tmp])
        Envy = Environment([deepcopy(y),tmp])
        initialize!(Envx)
        initialize!(Envy)
    end

    for i in 1:Nsweep
        totaltruncerror = 0
        @time "sweep $i finished, max truncation error = $(totaltruncerror)" begin
            println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α,map(z -> contract(z.envs[site].A, z.layer[1].ts[site:site+1]..., z.envs[site+2].A),[Envx,Envy])...)
                end; direction=:right,trunc = truncdim(D_MPO))
                map(z -> pushright!(z,map(DenseMPOTensor, [tl, tr])...),[Envx,Envy])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
            println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                tl, tr, temptruncerr = tsvd(let 
                    axpy!(α,map(z -> contract(z.envs[site-1].A, z.layer[1].ts[site-1:site]..., z.envs[site+1].A),[Envx,Envy])...)
                end; direction=:left,trunc = truncdim(D_MPO))
                map(z -> pushleft!(z,map(DenseMPOTensor, [tl, tr])...),[Envx,Envy])
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
        end
    end

    @assert Envx.layer[2] == Envy.layer[2]
    return axpy!(Envx.layer[2]',y)
end

function axpy!(α::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    return axpy!(α,1,x,y)
end

function axpy!(α::Number, β::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    return CompositeMPOTensor(x.A * α + y.A * β)
end

function axpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
    y.ts[:] = x.ts[:]
    return y
end

