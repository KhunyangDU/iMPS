function TensorKit.leftorth(elm::MPSTensor{3})
    return leftorth(elm.A,(1,2),(3,))
end

function TensorKit.leftorth(A::MPSTensor{R}) where R
    @assert R > 3
    Q,Rm = leftorth(A.A,(1,2),tuple(3:R...))
    return map(MPSTensor,(Q,permute(Rm,(1,),tuple(2:R-1...))))
end

function TensorKit.leftorth(A::MPSTensor{3}, B::MPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3] ≔ Rm[-1,1]*B.A[1,-2,-3]
    return map(MPSTensor,[Q,tmp])
end

function TensorKit.leftorth!(obj::DenseMPS,site::Int64)
    obj.ts[site:site+1] = leftorth(obj.ts[site:site+1]...)
end

function TensorKit.rightorth(A::MPSTensor{3})
    return rightorth(A.A,(1,),(2,3))
end

function TensorKit.rightorth(A::MPSTensor{R}) where R
    @assert R > 3
    Lm,Q = rightorth(A.A,(1,2),tuple(3:R...))
    return map(MPSTensor,(Lm, permute(Q,(1,),tuple(2:R-1...))))
end

function TensorKit.rightorth(A::MPSTensor{3}, B::MPSTensor{3})
    Lm,Q = rightorth(B)
    return map(MPSTensor,[A.A*Lm,permute(Q,(1,2),(3,))])
end

function TensorKit.rightorth!(obj::DenseMPS,site::Int64)
    obj.ts[site-1:site] = rightorth(obj.ts[site-1:site]...)
end



function TensorKit.tsvd(A::CompositeMPSTensor{2, R}; direction::Symbol=:center, kwargs...) where {R}
    @assert direction in [:center,:left,:right]
    U,S,V,ϵ = tsvd(A.A,(1,2),tuple(3:R...);kwargs...)
    if direction == :center
        return U,S,V,ϵ
    elseif direction == :left 
        return U*S,permute(V,(1,2),tuple(3:(R-1)...)),ϵ
    elseif direction == :right 
        return U,permute(S*V,(1,2),tuple(3:(R-1)...)),ϵ
    end
end


function canonicalize!(obj::DenseMPS{L,T},sl::Int64,sr::Int64) where {L,T}
    @assert 1 ≤ sl ≤ sr ≤ L 

    for sli in obj.center[1]:sl-1
        leftorth!(obj,sli)
        obj.center[1] += 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[2] += 1 )
    end
    for sri in obj.center[2]:-1:sr+1
        rightorth!(obj,sri)
        obj.center[2] -= 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[1] -= 1 )
    end
end


function canonicalize!(obj::DenseMPS{L,T},si::Int64) where {L,T}
    @assert 1 ≤ si ≤ L 
    canonicalize!(obj,si,si)
end
