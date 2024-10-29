

function groundEig(O::SparseProjectiveHamiltonian{N}, LanczosLevel::Int64) where N
    T, Q = Lanczos(O,_initialMPS(O),LanczosLevel)
    λ, v = eigen(T)
    Eg,Ev = argmin(real.(λ)) |> x -> (real.(λ)[x], sum(v[:, x] .* Q))
    return Eg, Ev / norm(Ev)
end

function _initialMPS(O::SparseProjectiveHamiltonian{1})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.envt[1].t, O.H.Mats[1].m[1,1].t])...)
    dom = collect(codomain(O.EnvR.envt[1].t))[1]
    return MPSTensor(randn,codom,dom)
end

function _initialMPS(O::SparseProjectiveHamiltonian{2})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.envt[1].t, [O.H.Mats[i].m[1,1].t for i in 1:2]...])...)
    dom = collect(codomain(O.EnvR.envt[1].t))[1]
    return CompositeMPSTensor(randn,codom,dom)
end

function Lanczos(O::SparseProjectiveHamiltonian{N}, q1::AbstractMPSTensor,
    LanczosLevel::Int64;kwargs...) where N
    Q = Vector{AbstractMPSTensor}(undef, LanczosLevel)
    α = zeros(LanczosLevel)
    β = zeros(LanczosLevel-1)

    Q[1] = q1

    for j = 1:LanczosLevel
        if j == 1
            w = action(O, Q[j])
        else
            w = action(O, Q[j]) - β[j-1] * Q[j-1]
        end

        α[j] = ApproxReal((w*adjoint(Q[j]))[1])
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


