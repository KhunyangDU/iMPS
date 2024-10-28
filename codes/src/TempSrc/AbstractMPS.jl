



mutable struct MPS{L, T<:Union{Float64, ComplexF64}} <: DenseMPS{L,T}
    Elements::Vector{MPSTensor}
    center::Vector{Int64}

    function MPS{L,T}(Elms::Vector{MPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(Elms,ct)
    end

    function MPS{L,T}(Elms::Vector{MPSTensor}) where {L,T}
        return new{L,T}(Elms,[1,L])
    end

    function MPS{L,T}(Elms::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([MPSTensor(elm) for elm in Elms],[1,L])
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: DenseMPS{L,T}
    Elements::Vector{AdjointMPSTensor}
    center::Vector{Int64}

    function AdjointMPS{L,T}(Elms::Vector{AdjointMPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(Elms,ct)
    end

    function AdjointMPS{L,T}(Elms::Vector{AdjointMPSTensor}) where {L,T}
        return new{L,T}(Elms,[1,length(Elms)])
    end

    function AdjointMPS{L,T}(Elms::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([AdjointMPSTensor(elm) for elm in Elms],[1,length(Elms)])
    end

end

function Base.length(::DenseMPS{L,T}) where {L,T}
    return L
end

function Base.adjoint(A::MPS{L,T}) where {L,T}
    return AdjointMPS{L,T}(adjoint(A.Elements), A.center)
end

"""
Generate rand MPS for initial state.

"""
function randMPS(PhySpaces::Vector,AuxSpaces::Vector;type::Type = Float64)
    @assert (L = length(PhySpaces)) == length(AuxSpaces)
    push!(AuxSpaces,trivial(PhySpaces[1]))
    tmp = Vector{MPSTensor}(undef,L)
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

function trivial(::GradedSpace{I, D}) where {I, D}
 
    dims = TensorKit.SortedVectorDict(one(I) => 1)
    return GradedSpace{I,D}(dims, false)
end
