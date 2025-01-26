function SETTN!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
    max_order = get(kwargs,:max_order,6)
    D = get(kwargs,:D,maximum(vcat(collect.(H.D)...)))
    F_tol = get(kwargs,:F_tol,1e-8)
    F = zeros(max_order)

    Hn = deepcopy(ρ)
    dF = 2*F_tol # make sure dF > F_tol
    for i in 1:max_order 
        Hn = mul!(Hn,deepcopy(Hn),H,1.,0.; D = D)
        ρ = axpy!((-β)^i / factorial(i),Hn ,ρ ; D = D)

        F[i] = - log(tr(ρ)) / β
        i ≠ 1 && (dF = abs((F[i] - F[i-1]) / F[i]))

        if dF < F_tol
           println("SETTN converged at $(i)th order with dF = $(dF)")
           break
        end

        i == max_order && println("SETTN not converged at max $(i)th order with dF = $(dF)") 
    end

    return ρ
end

