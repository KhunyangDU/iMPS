



mutable struct MPS{L, T<:Union{Float64, ComplexF64}} <: DenseMPS{L,T}
    Elements::Vector{AbstractMPSTensor}
    center::Vector{Int64}
    Const::T

    function MPS{L,T}(Elms::Vector{AbstractMPSTensor},
        ct::Vector{Int64},
        c::T = one(T)) where {L,T}
        return new{L,T}(Elms,ct,c)
    end

    function MPS{L,T}(Elms::Vector{AbstractMPSTensor}) where {L,T}
        return MPS{L,T}(Elms,[1,L])
    end

    function MPS{L,T}(Elms::Vector{AbstractTensorMap}) where {L,T}
        return MPS{L,T}([MPSTensor(elm) for elm in Elms],[1,L])
    end

end

function Base.length(mps::DenseMPS{L,T}) where {L,T}
    return L
end

"""
Generate rand MPS for initial state.

"""
function randMPS(PhySpaces::Vector,AuxSpaces::Vector;type::Type = Float64)
    @assert (L = length(PhySpaces)) == length(AuxSpaces)
    push!(AuxSpaces,AuxSpaces[1])
    tmp = Vector{AbstractMPSTensor}(undef,L)
    for i in 1:L
        tmp[i] = MPSTensor(randn,AuxSpaces[i] ⊗ PhySpaces[i],AuxSpaces[i+1])
    end
    return MPS{L,type}(tmp)
end


function randMPS(PhySpace::IndexSpace,AuxSpaces::Vector)
    return randMPS([PhySpace for i in eachindex(AuxSpaces)],AuxSpaces)
end


function getPhySpace(t::MPS)
    return getPhySpace(t.Elements[1])
end

function getPhySpace(t::MPSTensor{R}) where R
    if 3 ≤ R
        return codomain(t.Elements)[2]
    else
        return nothing
    end
end

function getAuxSpace(t::MPS)
    return getAuxSpace(t.Elements[1])
end

function getAuxSpace(t::MPSTensor)
    return codomain(t.Elements)[1]
end
