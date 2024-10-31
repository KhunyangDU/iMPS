abstract type AbstractObservableForest end

struct ObserableTree{N} <: AbstractObservableForest
    Root::InteractionTreeNode
    function ObserableTree{N}() where N
         
         Root = InteractionTreeNode(nothing)
         for i in 1:N
              addchild!(Root, InteractionTreeNode(IdentityOperator(0)))
         end
         return new{N}(Root)
    end

    ObserableTree() = ObserableTree{0}()
end

struct ObserableForest{N} <: AbstractObservableForest
    Roots::InteractionTreeNode
    function ObserableForest{N}() where N
         
         Root = InteractionTreeNode(nothing)
         for i in 1:N
              addchild!(Root, InteractionTreeNode(IdentityOperator(0)))
         end
         return new{N}(Root)
    end

    ObserableForest() = ObserableForest{0}()
end

