function TensorKit.leftorth(elm::MPSTensor{R}) where R 
    @assert 3 ≤ R
    Q,Rm = leftorth(elm.Elements,(1,2),tuple(3:R...))
    return [Q,Rm]
end

function TensorKit.leftorth!(obj::DenseMPS,site::Int64)
    obj.Elements[site].Elements,Rm = leftorth(obj.Elements[site])
    @tensor tmp[-1 -2;-3] ≔ Rm[-1,1]*obj.Elements[site+1].Elements[1,-2,-3]
    obj.Elements[site+1].Elements = tmp
end

function TensorKit.rightorth!(obj::DenseMPS,site::Int64)
    Lm,Q = rightorth(obj.Elements[site].Elements,(1,),(2,3))
    obj.Elements[site-1:site] = map(MPSTensor,[obj.Elements[site-1].Elements*Lm,permute(Q,(1,2),(3,))])
end



function canonicalize!(obj::DenseMPS{L,T},sl::Int64,sr::Int64) where {L,T}
    @assert 1 ≤ sl ≤ sr ≤ L 

    for sli in obj.center[1]:sl-1
        leftorth!(obj,sli)
        obj.center[1] += 1
    end
    for sri in obj.center[2]:-1:sr+1
        rightorth!(obj,sri)
        obj.center[2] -= 1
    end
end


function canonicalize!(obj::DenseMPS{L,T},si::Int64) where {L,T}
    @assert 1 ≤ si ≤ L 
    canonicalize!(obj,si,si)
end
