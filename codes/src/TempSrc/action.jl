function action(O::SparseProjectiveHamiltonian{N},obj::AbstractMPSTensor{R}) where {N,R}
    if (N,R) == (1,3)
        return action1(O,obj)
    elseif (N,R) == (2,4) 
        return action2(O,obj)
    end
end

function action1(O::SparseProjectiveHamiltonian{1},obj::MPSTensor{3})
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
end


function action2(O::SparseProjectiveHamiltonian{2},obj::CompositeMPSTensor{2,4})
    N,M = O.H.D[1]

    tmp1 = Vector{Any}(nothing,M)

    for i in 1:M, j in 1:N
        isnothing(O.H.Mats[1].m[j,i]) && continue
        if isnothing(tmp1[i])
            tmp1[i] = contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
        else 
            tmp1[i] += contract(O.EnvL.envt[j], obj, O.H.Mats[1].m[j,i])
        end
    end

    ref = obj.A * 0

    N,M = O.H.D[2]

    for i in 1:M
        tmp2 = nothing
        for j in 1:N
            isnothing(O.H.Mats[2].m[j,i]) && continue
            if isnothing(tmp2)
                tmp2 = contract(tmp1[j], O.H.Mats[2].m[j,i])
            else 
                tmp2 += contract(tmp1[j], O.H.Mats[2].m[j,i])
            end
        end
        ref += contract(tmp2,O.EnvR.envt[i])
    end

    return CompositeMPSTensor(ref)
end


