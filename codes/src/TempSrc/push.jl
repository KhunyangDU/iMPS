function pushleft!(env::Environment)
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushleft(mps::MPS, mpo::SparseMPO, EnvR::SparseEnvironmentTensor, site::Int64)
    @assert mpo.D[site][2] == EnvR.D
    tmpEnvR = Vector{Any}(nothing,mpo.D[site][1])
    for i in eachindex(tmpEnvR), j in 1:EnvR.D
        isnothing(mpo.Mats[site][i,j]) && continue
        if isnothing(tmpEnvR[i])
            tmpEnvR[i] = contract(mps.Elements[site],mpo.Mats[site][i,j],EnvR.envt[j])
        else 
            tmpEnvR[i] += contract(mps.Elements[site],mpo.Mats[site][i,j],EnvR.envt[j])
        end
    end

    return SparseEnvironmentTensor(convert(Vector{AbstractEnvironmentTensor},tmpEnvR))
end


function pushright!(env::Environment)
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

function pushright(mps::MPS, mpo::SparseMPO, EnvL::SparseEnvironmentTensor, site::Int64)
    @assert mpo.D[site][1] == EnvL.D
    tmpEnvL = Vector{Any}(nothing,mpo.D[site][2])
    for i in eachindex(tmpEnvL), j in 1:EnvL.D
        isnothing(mpo.Mats[site][j,i]) && continue
        if isnothing(tmpEnvL[i])
            tmpEnvL[i] = contract(mps.Elements[site],mpo.Mats[site][j,i],EnvL.envt[j])
        else 
            tmpEnvL[i] += contract(mps.Elements[site],mpo.Mats[site][j,i],EnvL.envt[j])
        end
    end

    return SparseEnvironmentTensor(convert(Vector{AbstractEnvironmentTensor},tmpEnvL))

end


function contract(mpst::MPSTensor{3},mpot::MPOTensor{2},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ mpst.Elements[-1,4,1] * mpot.t[3,4] * mpst.Elements'[2,-2,3] * EnvR.t[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ mpst.Elements[-1,4,1] * mpot.t[3,-2,4] * mpst.Elements'[2,-3,3] * EnvR.t[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{2},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ mpst.Elements[-1,4,1] * mpot.t[3,4] * mpst.Elements'[2,-3,3] * EnvR.t[1,-2,2]
    return RightEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ mpst.Elements[-1,4,1] * mpot.t[3,4,5] * mpst.Elements'[2,-2,3] * EnvR.t[1,5,2]
    return RightEnvironmentTensor(tmp)
end

# ==============

function contract(mpst::MPSTensor{3},mpot::MPOTensor{2},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ mpst.Elements[2,4,-2] * mpot.t[3,4] * mpst.Elements'[-1,1,3] * EnvL.t[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2 -3] ≔ mpst.Elements[2,4,-3] * mpot.t[3,4,-2] * mpst.Elements'[-1,1,3] * EnvL.t[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{2},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ mpst.Elements[2,4,-3] * mpot.t[3,4] * mpst.Elements'[-1,1,3] * EnvL.t[1,-2,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(mpst::MPSTensor{3},mpot::MPOTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ mpst.Elements[2,4,-2] * mpot.t[3,5,4] * mpst.Elements'[-1,1,3] * EnvL.t[1,5,2]
    return LeftEnvironmentTensor(tmp)
end

