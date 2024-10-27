




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

abstract type AbstractMPOTensor end

mutable struct MPOTensor{R} <: AbstractMPOTensor
    t::AbstractTensorMap

    function MPOTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

function Base.length(::MPOTensor)
    return 1
end

function Base.iterate(t::MPOTensor)
    return (t,nothing)
end

function Base.iterate(::MPOTensor,::Nothing)
    return nothing
end

abstract type AbstractMPS{L} end
abstract type DenseMPS{L, T <:Union{Float64, ComplexF64}} <: AbstractMPS{L} end


abstract type AbstractMPO end


function rank(A::AbstractTensorMap)
    return length(codomain(A)) + length(domain(A))
end

