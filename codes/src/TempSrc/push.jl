function pushleft!(env::Environment{3})
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushleft(A::DenseMPS, mpo::SparseMPO, B::DenseMPS, EnvR::SparseRightEnvironmentTensor, site::Int64)
    @assert mpo.D[site][2] == EnvR.D
    tmpEnvR = Vector{Any}(nothing,mpo.D[site][1])
    for i in eachindex(tmpEnvR), j in 1:EnvR.D
        isnothing(mpo.Mats[site].m[i,j]) && continue
        if isnothing(tmpEnvR[i])
            tmpEnvR[i] = contract(A.Elements[site], mpo.Mats[site].m[i,j], B.Elements[site], EnvR.envt[j])
        else 
            tmpEnvR[i] += contract(A.Elements[site], mpo.Mats[site].m[i,j], B.Elements[site], EnvR.envt[j])
        end
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor},tmpEnvR))
end


function pushright!(env::Environment{3})
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

function pushright(A::DenseMPS, mpo::SparseMPO, B::DenseMPS, EnvL::SparseLeftEnvironmentTensor, site::Int64)
    @assert mpo.D[site][1] == EnvL.D
    tmpEnvL = Vector{Any}(nothing,mpo.D[site][2])
    for i in eachindex(tmpEnvL), j in 1:EnvL.D
        isnothing(mpo.Mats[site].m[j,i]) && continue
        if isnothing(tmpEnvL[i])
            tmpEnvL[i] = contract(A.Elements[site], mpo.Mats[site].m[j,i], B.Elements[site],EnvL.envt[j])
        else 
            tmpEnvL[i] += contract(A.Elements[site], mpo.Mats[site].m[j,i], B.Elements[site],EnvL.envt[j])
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor},tmpEnvL))
end


function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.Elements[-1,4,1] * mpot.t[3,4] * B.Elements[2,-2,3] * EnvR.t[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.Elements[-1,4,1] * mpot.t[3,-2,4] * B.Elements[2,-3,3] * EnvR.t[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.Elements[-1,4,1] * mpot.t[3,4] * B.Elements[2,-3,3] * EnvR.t[1,-2,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.Elements[-1,4,1] * mpot.t[3,5,4] * B.Elements[2,-2,3] * EnvR.t[1,5,2]
    return RightEnvironmentTensor(tmp)
end

# ==============

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.Elements[2,4,-2] * mpot.t[3,4] * B.Elements[-1,1,3] * EnvL.t[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2 -3] ≔ A.Elements[2,4,-3] * mpot.t[3,-2,4] * B.Elements[-1,1,3] * EnvL.t[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.Elements[2,4,-3] * mpot.t[3,4] * B.Elements[-1,1,3] * EnvL.t[1,-2,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.Elements[2,4,-2] * mpot.t[3,5,4] * B.Elements[-1,1,3] * EnvL.t[1,5,2]
    return LeftEnvironmentTensor(tmp)
end

