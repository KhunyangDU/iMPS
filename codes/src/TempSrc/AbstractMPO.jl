


mutable struct SparseMPO <: AbstractMPO
    Mats::Vector{Matrix{Union{Nothing,AbstractMPOTensor}}}
    D::Vector{NTuple{2,Int64}}
    
    function SparseMPO(Mats::Vector{Matrix{Union{Nothing,AbstractMPOTensor}}},
        D::Vector{NTuple{2,Int64}})
        return new(Mats,D)
    end

    function SparseMPO(Mats::Vector{Matrix{Union{Nothing,AbstractMPOTensor}}})
        D = map(size,Mats)
        return new(Mats,convert(Vector{NTuple{2,Int64}},D))
    end

end


function issparse(::SparseMPO)
    return true
end



function AutomataSparseMPO(Root::InteractionTreeNode,L::Int64=treeheight(Root) - 1)
    MPO = let 
        tempMPO = Vector{Matrix{Union{Nothing,AbstractMPOTensor}}}(undef,L)

        last_leaves = []
        last_roots = Root.children

        idtensor = getIdTensor(last_roots[1].children[1].Opr)
        
        last_inverse_root = 0
        next_inverse_root = 0
        
        for iL in 1:L

            if next_inverse_root == 0 && !isempty(findall(x -> isempty(x.children),vcat([lastroot.children for lastroot in last_roots]...)))
                next_inverse_root = 1
            end

            next_leaves = []
            leaves_inds = []

            next_roots = []
            roots_inds = []

            for (lastind,last_root) in enumerate(last_roots)
                for next_subtree in last_root.children
                    if isempty(next_subtree.children)
                        push!(next_leaves,next_subtree)
                        push!(leaves_inds,(lastind + last_inverse_root, next_inverse_root, length(next_leaves)))
                    else
                        push!(next_roots,next_subtree)
                        push!(roots_inds,(lastind + last_inverse_root, length(next_roots) + next_inverse_root, length(next_roots)))
                    end
                end
            end

            localMPOdims = length.((last_roots,next_roots)) .+ (last_inverse_root,next_inverse_root)
            localMPO = Matrix{Union{Nothing,AbstractMPOTensor}}(nothing,localMPOdims...)
            #localMPO .= MPOTensor(0*idtensor)
            localMPO[1,1] = MPOTensor(last_inverse_root*idtensor)


            for inds in leaves_inds
                localMPO[inds[1:2]...] = MPOTensor(let 
                    localOpr = next_leaves[inds[3]].Opr.Opri
                    strength = next_leaves[inds[3]].Opr.strength
                    if isnan(strength)
                        localOpr
                    else
                        localOpr*strength
                    end
                end)
            end

            for inds in roots_inds
                localMPO[inds[1:2]...] = MPOTensor(let 
                    localOpr = next_roots[inds[3]].Opr.Opri
                    strength = next_roots[inds[3]].Opr.strength
                    if isnan(strength)
                        localOpr
                    else
                        localOpr*strength
                    end
                end)
            end
            
            last_leaves = next_leaves
            last_roots = next_roots
            last_inverse_root = next_inverse_root

            tempMPO[iL] = localMPO
        end

        SparseMPO(tempMPO)
    end

    return MPO
end

function AutomataSparseMPO(Tree::InteractionTree,L::Int64 = treeheight(Tree.Root) - 1)
    return AutomataSparseMPO(Tree.Root,L)
end


