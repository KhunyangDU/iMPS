
abstract type AbstractHamiltonian end
abstract type AbstractProjectiveHamiltonian <: AbstractHamiltonian end
mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::SparseMPO

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor,
        H::SparseMPO) 
        return new{length(H.Mats)}(EnvL,EnvR,H)
    end
end

function project1(H::SparseMPO,env::Environment,site::Int64)
    return SparseProjectiveHamiltonian(env.envs[site:site+1]...,SparseMPO(H.Mats[site]))
end

function project2(H::SparseMPO,env::Environment,site::Int64)
    return SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(H.Mats[site:site+1]))
end



