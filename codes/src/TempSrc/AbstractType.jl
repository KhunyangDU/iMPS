




abstract type AbstractMPSTensor{R} end
"""
     struct MPSTensor <: AbstractMPSTensor
          A::AbstractTensorMap
     end 
          
Wrapper type for MPS local tensors.

Convention (' marks codomain): 

          3 ... (R-1)
          \\ | /  
     1'--   A  ---R         1'-- A -- 2
            | 
            2'   
In particular, R == 2 for bond tensor.

# Constructors
     MPSTensor(::AbstractTensorMap) 
"""
mutable struct MPSTensor{R} <: AbstractMPSTensor{R}
    Elements::AbstractTensorMap

    function MPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function MPSTensor(fc::Function,codomain,domain)
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

end

mutable struct AdjointMPSTensor{R} <: AbstractMPSTensor{R}
    Elements::AbstractTensorMap

    function AdjointMPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function AdjointMPSTensor(fc::Function,codomain,domain)
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

end

function Base.adjoint(t::MPSTensor)
    return AdjointMPSTensor(t.Elements')
end

function Base.adjoint(ts::Vector{MPSTensor})
    return convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(t.Elements') for t in ts])
end

function Base.adjoint(t::AdjointMPSTensor)
    return MPSTensor(t.Elements')
end

function Base.adjoint(ts::Vector{AdjointMPSTensor})
    return convert(Vector{MPSTensor},[MPSTensor(t.Elements') for t in ts])
end


function Base.:*(A::MPSTensor{3}, Ad::AdjointMPSTensor{3})
    return @tensor A.Elements[1,2,3] * Ad.Elements[3,1,2]
end

function Base.:*(n::Number, A::MPSTensor)
    return MPSTensor(A.Elements*n)
end

function Base.:*(n::Number, A::MPSTensor)
    return MPSTensor(A.Elements*n)
end

function Base.:/(A::MPSTensor, n::Number)
    @assert n ≠ 0
    return (1/n)*A
end

function Base.:+(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
    @assert R₁ == R₂
    return MPSTensor(A.Elements + B.Elements)
end

function Base.:-(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
    return A + (-1)*B
end

function TensorKit.norm(A::MPSTensor)
    return norm(A.Elements)
end

mutable struct CompositeMPSTensor{R} <: AbstractMPSTensor{R}
    A::AbstractTensorMap

    function CompositeMPSTensor(A::AbstractTensorMap)
        return new{rank(A)}(A)
    end

    function CompositeMPSTensor(fc::Function,codomain,domain)
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end
end

function composite(A::MPSTensor{3}, B::MPSTensor{3})
    @tensor tmp[-1 -2 -3; -4] ≔ A.Elements[-1,-2,1]*B.Elements[1,-3,-4]
    return CompositeMPSTensor(tmp)
end

mutable struct AdjointCompositeMPSTensor{R} <: AbstractMPSTensor{R}
    A::AbstractTensorMap

    function AdjointCompositeMPSTensor(A::AbstractTensorMap)
        return new{rank(A)}(A)
    end

    function AdjointCompositeMPSTensor(fc::Function,codomain,domain)
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end
end

function Base.adjoint(t::CompositeMPSTensor)
    return AdjointCompositeMPSTensor(t.A)
end

function Base.adjoint(ts::Vector{CompositeMPSTensor})
    return convert(Vector{AdjointCompositeMPSTensor},[AdjointCompositeMPSTensor(t.A) for t in ts])
end

function Base.adjoint(t::AdjointCompositeMPSTensor)
    return CompositeMPSTensor(t.A)
end

function Base.adjoint(ts::Vector{AdjointCompositeMPSTensor})
    return convert(Vector{CompositeMPSTensor},[CompositeMPSTensor(t.A) for t in ts])
end

abstract type AbstractMPOTensor end

mutable struct DenseMPOTensor{R} <: AbstractMPOTensor
    t::AbstractTensorMap

    function DenseMPOTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

mutable struct SparseMPOTensor{N,M} <: AbstractMPOTensor
    m::Matrix{Union{Nothing,DenseMPOTensor}}

    function SparseMPOTensor(m::Matrix{Union{Nothing,DenseMPOTensor}})
        return new{size(m)...}(m::Matrix{Union{Nothing,DenseMPOTensor}})
    end

    function SparseMPOTensor(::Nothing,N::Int64,M::Int64)
        return new{N,M}(Matrix{Union{Nothing,DenseMPOTensor}}(nothing,N,M))
    end
end

function Base.size(::SparseMPOTensor{N,M}) where {N,M}
    return N,M
end



function Base.length(::DenseMPOTensor)
    return 1
end

function Base.iterate(t::DenseMPOTensor)
    return (t,nothing)
end

function Base.iterate(::DenseMPOTensor,::Nothing)
    return nothing
end

abstract type AbstractMPS end
abstract type DenseMPS{L,T} <: AbstractMPS end


abstract type AbstractMPO end


function rank(A::AbstractTensorMap)
    return length(codomain(A)) + length(domain(A))
end

