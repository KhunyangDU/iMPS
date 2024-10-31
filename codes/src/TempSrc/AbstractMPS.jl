



mutable struct MPS{L, T<:Union{Float64, ComplexF64}} <: DenseMPS{L,T}
    ts::Vector{MPSTensor}
    center::Vector{Int64}

    function MPS{L,T}(ts::Vector{MPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function MPS{L,T}(ts::Vector{MPSTensor}) where {L,T}
        return new{L,T}(ts,[1,L])
    end

    function MPS{L,T}(ts::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([MPSTensor(t) for t in ts],[1,L])
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: DenseMPS{L,T}
    ts::Vector{AdjointMPSTensor}
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
    return AdjointMPS{L,T}(adjoint(A.ts), A.center)
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

    obj = MPS{L,type}(tmp)

    canonicalize!(obj, Lx)
    canonicalize!(obj, 1)
    normalize!(obj)

    return obj
end


function randMPS(PhySpace::IndexSpace,AuxSpaces::Vector)
    return randMPS([PhySpace for i in eachindex(AuxSpaces)],AuxSpaces)
end


function getPhySpace(t::MPS)
    return getPhySpace(t.ts[1])
end

function getPhySpace(t::MPSTensor{R}) where R
    if 3 ≤ R
        return codomain(t.A)[2]
    else
        return nothing
    end
end

function getAuxSpace(t::MPS)
    return getAuxSpace(t.ts[1])
end

function getAuxSpace(t::MPSTensor)
    return codomain(t.A)[1]
end

function trivial(::GradedSpace{I, D}) where {I, D}
    dims = TensorKit.SortedVectorDict(one(I) => 1)
    return GradedSpace{I,D}(dims, false)
end

function trivial(::ComplexSpace)
    return ℂ^1
end

function normalize!(obj::DenseMPS{L,T}) where {L,T}
    @assert 1 == obj.center[1] == obj.center[2]
    obj.ts[1] /= norm(obj.ts[1])
end

function normalize!(obj::AbstractMPSTensor)
    obj.A = obj.A / norm(obj.A)
end
