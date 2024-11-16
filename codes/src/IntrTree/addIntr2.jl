function addIntr2!(
    Root::InteractionTreeNode,
    Opris::NTuple{2,AbstractTensorMap},
    sites::NTuple{2,Int64},
    names::NTuple{2,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    OprL,OprR = map(x -> LocalOperator(Opris[x],names[x],sites[x]),1:2)
    addIntr2!(Root,OprL,OprR,strength,Z)
end

function addIntr2!(
    Root::InteractionTreeNode,
    OprL::LocalOperator,OprR::LocalOperator,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    @assert OprL.site < OprR.site

    current_node = Root
    current_site = 1

    # add the identity
    while current_site < OprR.site

        tempOpr = let 
            if current_site < OprL.site
                IdentityOperator(getIdTensor(OprL),current_site)
            elseif current_site == OprL.site
                OprL
            elseif isnothing(Z)
                IdentityOperator(getIdTensor(OprL),current_site)
            else
                LocalOperator(Z,"Z",current_site)
            end
        end

        indId = findfirst(x -> isequal(x.Opr,tempOpr),current_node.children)
        if isnothing(indId)
            addchild!(current_node,tempOpr)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

        current_site += 1
    end

    # add the right Opr
    if !isnothing(Z)
        _addZ!(OprR,Z)
    end

    addchild!(current_node, OprR)
    current_node.children[end].Opr.strength = strength
end

function _addZ!(OprR::LocalOperator, Z::AbstractTensorMap)
    OprR.Opri = _addZ(OprR.Opri,Z)
    OprR.name = string("Z",OprR.name)
end

function _addZ(Opri::AbstractTensorMap{S₁,2,1}, Z::AbstractTensorMap{S₂,1,1}) where {S₁,S₂}
    @tensor tmp[-1 -2;-3] ≔ Z[-1,1] * Opri[1,-2,-3]
    return tmp
end

function _addZ(Opri::AbstractTensorMap{S₁,1,1}, Z::AbstractTensorMap{S₂,1,1}) where {S₁,S₂}
    @tensor tmp[-1;-2] ≔ Z[-1,1] * Opri[1,-2]
    return tmp
end

