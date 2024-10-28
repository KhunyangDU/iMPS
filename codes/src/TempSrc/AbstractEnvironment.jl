
abstract type AbstractEnvironmentTensor end
abstract type AbstractLeftEnvironmentTensor <: AbstractEnvironmentTensor end
abstract type AbstractRightEnvironmentTensor <: AbstractEnvironmentTensor end

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

mutable struct LeftCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    t::AbstractTensorMap

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(codomain(t)),rank(t)}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    t::AbstractTensorMap

    function RightCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(domain(t)),rank(t)}(t)
    end
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



mutable struct SparseLeftEnvironmentTensor <: AbstractLeftEnvironmentTensor
    envt::Vector{LeftEnvironmentTensor}
    D::Int64

    function SparseLeftEnvironmentTensor(t::Vector{LeftEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseLeftEnvironmentTensor(t::Vector{LeftEnvironmentTensor})
        return new(t,length(t))
    end

    function SparseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new(convert(Vector{LeftEnvironmentTensor},[t]),1)
    end

    function SparseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new(convert(Vector{LeftEnvironmentTensor},[LeftEnvironmentTensor(t)]),1)
    end

    function SparseLeftEnvironmentTensor(t::Vector{AbstractTensorMap})
        return new(convert(Vector{LeftEnvironmentTensor},[LeftEnvironmentTensor(ti) for ti in t]),length(t))
    end
end

mutable struct SparseRightEnvironmentTensor <: AbstractRightEnvironmentTensor
    envt::Vector{RightEnvironmentTensor}
    D::Int64

    function SparseRightEnvironmentTensor(t::Vector{RightEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseRightEnvironmentTensor(t::Vector{RightEnvironmentTensor})
        return new(t,length(t))
    end

    function SparseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new(convert(Vector{RightEnvironmentTensor},[t]),1)
    end

    function SparseRightEnvironmentTensor(t::AbstractTensorMap)
        return new(convert(Vector{RightEnvironmentTensor},[RightEnvironmentTensor(t)]),1)
    end

    function SparseRightEnvironmentTensor(t::Vector{AbstractTensorMap})
        return new(convert(Vector{RightEnvironmentTensor},[RightEnvironmentTensor(ti) for ti in t]),length(t))
    end
end



abstract type AbstractEnvironment end
"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment{N} <: AbstractEnvironment
    layer::Vector
    envs::Union{Nothing,Vector{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64

    function Environment(layer::Vector,
        envs::Vector{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64})
        return new{length(layer)}(layer,envs,center,L)
    end

    function Environment(layer::Vector)
        L = length(layer[1])
        return new{length(layer)}(layer,nothing,[1,L],L)
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

    for _ in env.center[2]:-1:sr+1
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
        env.envs[1] = SparseLeftEnvironmentTensor(tmp)
        env.envs[end] = SparseRightEnvironmentTensor(tmp)
    end
end
