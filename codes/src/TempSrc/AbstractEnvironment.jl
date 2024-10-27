
abstract type AbstractEnvironmentTensor end

mutable struct LocalEnvironmentTensor{R} <: AbstractEnvironmentTensor
    t::AbstractTensorMap

    function LocalEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

mutable struct RightEnvironmentTensor{R} <: AbstractEnvironmentTensor
    t::AbstractTensorMap

    function RightEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end



mutable struct LeftEnvironmentTensor{R} <: AbstractEnvironmentTensor
    t::AbstractTensorMap

    function LeftEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

function Base.:+(A::LeftEnvironmentTensor,
    B::LeftEnvironmentTensor)
    return LeftEnvironmentTensor(A.t + B.t)
end

function Base.:+(A::RightEnvironmentTensor,
    B::RightEnvironmentTensor)
    return RightEnvironmentTensor(A.t + B.t)
end


mutable struct SparseEnvironmentTensor <: AbstractEnvironmentTensor
    envt::Vector{AbstractEnvironmentTensor}
    D::Int64

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor})
        return new(t,length(t))
    end

    function SparseEnvironmentTensor(t::Union{LeftEnvironmentTensor,RightEnvironmentTensor})
        return new(convert(Vector{AbstractEnvironmentTensor},[t]),1)
    end
end



abstract type AbstractEnvironment end
"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment <: AbstractEnvironment
    layer::Vector
    envs::Union{Nothing,Vector{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64

    function Environment(layer::Vector,
        envs::Vector{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64})
        return new(layer,envs,center,L)
    end

    function Environment(layer::Vector)
        L = length(layer[1])
        return new(layer,nothing,[1,L],L)
    end

end

function initialize!(env::Environment)
    env.envs = Vector{AbstractEnvironmentTensor}(undef, env.L + 1)
    setdefault!(env)
    canonicalize!(env,1)
end

function canonicalize!(env::Environment,sl::Int64,sr::Int64)
    @assert 1 ≤ sl ≤ sr ≤ env.L + 1

    for _ in env.center[1]:sl-1
        pushright!(env)
    end

    for i in env.center[2]:-1:sr+1
        pushleft!(env)
    end

end

function canonicalize!(env::Environment,si::Int64)
    @assert 1 ≤ si ≤ env.L + 1
    canonicalize!(env,si,si)
end


function setdefault!(env::Environment)
    if issparse(env.layer[2])
        AuxSpace = getAuxSpace(env.layer[1])
        tmp = isometry(AuxSpace,AuxSpace)
        env.envs[1] = SparseEnvironmentTensor(LeftEnvironmentTensor(tmp))
        env.envs[end] = SparseEnvironmentTensor(RightEnvironmentTensor(tmp))
    end
end
