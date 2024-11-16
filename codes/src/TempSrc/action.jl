function action(O::SparseProjectiveHamiltonian{N},obj::AbstractMPSTensor{R}) where {N,R}
    if (N,R) == (1,3)
        return action1(O,obj)
    elseif (N,R) == (2,4) 
        return action2(O,obj)
    end
end

#= function action1(O::SparseProjectiveHamiltonian{1},obj::MPSTensor{3})
    N,M = O.H.D[1]

    ref = obj.A * 0

    #tmp = Vector{Any}(nothing,M)
    for i in 1:M
        tmp = nothing
        for j in 1:N
            isnothing(O.H.Mats[1].m[j,i]) && continue
            if isnothing(tmp)
                tmp = contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
            else 
                tmp += contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
            end
        end
        ref += contract(tmp,O.EnvR.envt[i])
    end

    return MPSTensor(ref)
end =#


#= function action2(O::SparseProjectiveHamiltonian{2},obj::CompositeMPSTensor{2,4})
    N,M = O.H.D[1]
    @show N,M

    tmp1 = Vector{Any}(nothing,M)
    @show O.H.Mats[1].m[1,1],O.H.Mats[1].m[1,2]

    for i in 1:M, j in 1:N
        isnothing(O.H.Mats[1].m[j,i]) && continue
        if isnothing(tmp1[i])
            tmp1[i] = contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
        else 
            tmp1[i] += contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
        end
        @show i,space(tmp1[i].t)
    end

    ref = obj.A * 0

    N,M = O.H.D[2]

    for i in 1:M
        tmp2 = nothing
        for j in 1:N
            isnothing(O.H.Mats[2].m[j,i]) && continue
            @show i,j
            @show tmp1[j].t
            @show O.H.Mats[2].m[j,i].t
            @show O.EnvR.envt[i].t
            if isnothing(tmp2)
                tmp2 = contract(tmp1[j], O.H.Mats[2].m[j,i])
            else 
                tmp2 += contract(tmp1[j], O.H.Mats[2].m[j,i])
            end
            @show space(tmp2.t)
        end
        ref += contract(tmp2,O.EnvR.envt[i])
    end

    return CompositeMPSTensor(ref)
end =#

function action1(O::SparseProjectiveHamiltonian{1},obj::MPSTensor{3})
    N,M = O.H.D[1]
    ts = obj.A * 0

    for i in 1:N, j in 1:M
        isnothing(O.H.Mats[1].m[i,j]) && continue
        tmp = contract(O.EnvL.envt[i], obj, O.H.Mats[1].m[i,j])
        ts += contract(tmp,O.EnvR.envt[j])
    end

    return MPSTensor(ts)
end

function action2(O::SparseProjectiveHamiltonian{2},obj::CompositeMPSTensor{2,4})
    N,M1 = O.H.D[1]
    M2,R = O.H.D[2]
    @assert M1 == M2

    ts = obj.A * 0

    for i in 1:N, j in 1:M1, k in 1:R
        isnothing(O.H.Mats[1].m[i,j]) | isnothing(O.H.Mats[2].m[j,k]) && continue
        tmp1 = contract(O.EnvL.envt[i], obj, O.H.Mats[1].m[i,j])
        tmp2 = contract(tmp1, O.H.Mats[2].m[j,k])
        ts += contract(tmp2,O.EnvR.envt[k])
    end

    return CompositeMPSTensor(ts)
end

function contract(El::LeftCompositeEnvironmentTensor{3,5}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.t[-1,-2,1,-4,-5] * mpo.t[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

