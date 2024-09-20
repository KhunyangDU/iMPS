function Lanczos(M::AbstractTensorMap,level::Int64)
    Mdm = domain(M)
    Q = Vector{AbstractTensorMap}(undef, level)
    α = zeros(ComplexF64,level)
    β = zeros(ComplexF64,level-1)

    q1 = Tensor(rand, ComplexF64, ⊗([dm for dm in Mdm]...))
    q1 /= norm(q1)
    Q[1] = q1

    for j = 1:level
        if j == 1
            w = M * Q[j]
        else
            w = M * Q[j] - β[j-1] * Q[j-1]
        end

        
        α[j] = (Q[j]'*w)[1]
        w -= α[j] * Q[j]
        
        if j < level
            β[j] = norm(w)
            if β[j] ≈ 0
                @error "flow interrupted"
            else
                Q[j+1] = w / β[j]
            end
        end
    end
    
    T = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
    return T, Q
    
end

function groundEig(M::AbstractTensorMap, level::Int64,D_MPS::Int64)
    T, Q = Lanczos(M, level)
    λ, v = eigen(T)
    Eg,Ev = argmin(real.(λ)) |> x -> (real.(λ)[x], sum(Q .* v[:, x]))
    return Eg, Ev' / norm(Ev)
end

function Lanczos(A::AbstractMatrix, k::Int; q1::Vector = rand(size(A, 1)))
    n = size(A, 1)
    Q = zeros(ComplexF64, n, k)
    α = zeros(k)
    β = zeros(k-1)
    
    q1 /= norm(q1)
    Q[:, 1] = q1
    
    for j = 1:k
        if j == 1
            w = A * Q[:, j]
        else
            w = A * Q[:, j] - β[j-1] * Q[:, j-1]
        end
        
        α[j] = ApproxReal(dot(Q[:, j], w))
        w -= α[j] * Q[:, j]
        
        if j < k
            β[j] = norm(w)
            if β[j] ≈ 0
                ns = nullspace(Q[:,1:j])
                Q[:, j+1] = ns * randn(size(ns, 2))
            else
                Q[:, j+1] = w / β[j]
            end
        end
    end
    
    T = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
    return T, Q
end

function groundEig(A::AbstractMatrix, k::Int)
    T, Q = Lanczos(A, k)
    λ, v = eigen(T)
    return argmin(λ) |> x -> (λ[x], Q * v[:, x])
end

function Lanczos(Hi::Vector,
    EnvL::AbstractTensorMap,EnvR::AbstractTensorMap,
    LanczosLevel::Int64;kwargs...)
    Q = Vector{AbstractTensorMap}(undef, LanczosLevel)
    α = zeros(LanczosLevel)
    β = zeros(LanczosLevel-1)

    q1 = get(kwargs,:q1,let 
        Tensor(rand, ComplexF64, ⊗(codomain(EnvL)[1],let 
            spaces = []
            for h in Hi 
                push!(spaces,domain(h)[2])
            end
            spaces
            end...,codomain(EnvR)[1])) |> x -> x' / norm(x)
    end)
    Q[1] = q1


    for j = 1:LanczosLevel
        if j == 1
            w = LocalMerge(EnvL,Q[j],Hi...,EnvR)
        else
            w = LocalMerge(EnvL,Q[j],Hi...,EnvR) - β[j-1] * Q[j-1]
        end

        α[j] = ApproxReal((w*Q[j]')[1])
        w -= α[j] * Q[j]
        
        if j < LanczosLevel
            β[j] = norm(w)
            if β[j] ≈ 0
                @error "flow interrupted"
            else
                Q[j+1] = w / β[j]
            end
        end
    end
    
    T = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
    return T, Q
    
end

function groundEig(Hi::Vector{AbstractTensorMap{ComplexSpace,2,2}},
    EnvL::AbstractTensorMap,EnvR::AbstractTensorMap,
    LanczosLevel::Int64)
    T, Q = Lanczos(Hi,EnvL,EnvR,LanczosLevel)
    λ, v = eigen(T)
    Eg,Ev = argmin(real.(λ)) |> x -> (real.(λ)[x], sum(Q .* v[:, x]))
    return Eg, Ev / norm(Ev)
end

function Evolve(
    localψ::AbstractTensorMap,
    Hi::Vector{Union{AbstractTensorMap{ComplexSpace,1,3},AbstractTensorMap{ComplexSpace,2,2}}},
    EnvL::AbstractTensorMap,EnvR::AbstractTensorMap,
    τ::Number,
    LanczosLevel::Int64)
    τHi = Vector{Union{AbstractTensorMap{ComplexSpace,1,3},AbstractTensorMap{ComplexSpace,2,2}}}(undef,length(Hi))
    for (hi,h) in enumerate(Hi)
        if typeof(h) <: AbstractTensorMap{ComplexSpace,1,3}
            τHi[hi] = -τ*h
        else
            τHi[hi] = h
        end
    end
    T, Q = Lanczos(τHi,EnvL,EnvR,LanczosLevel;q1 = localψ / norm(localψ))
    A = sum(norm(localψ) * exp(T)[:,1] .* Q)
    return A
    #return A |> x -> x*norm(localψ)
end
