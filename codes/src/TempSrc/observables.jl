abstract type AbstractObservable end

mutable struct MPSObservable <: AbstractObservable
    forest::Union{Nothing, AbstractObservableForest}
    values::Union{Nothing, AbstractDict}
    L::Union{Nothing, Int64}

    function MPSObservable(forest::AbstractObservableForest,
        L::Int64)
        return new(forest, L) 
    end

    function MPSObservable(forest::AbstractObservableForest)
        return new(forest, treeheight(obj.forest.Roots) - 2) 
    end

    MPSObservable() = new(ObserableForest(), nothing)
end

function update!(obj::MPSObservable)
    obj.L = treeheight(obj.forest.Roots) - 2
end

function addObs!(Obs::MPSObservable,     
    Opri::Union{AbstractTensorMap,NTuple{2,AbstractTensorMap}},
    site::Union{Int64,NTuple{2,Int64}},
    name::Union{String,NTuple{2,String}},
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

function addObs!(Obs::MPSObservable,     
    Opri::AbstractTensorMap,
    site::Int64,
    name::String,
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

function addObs!(Obs::MPSObservable,     
    Opri::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

function calObs!(Obs::MPSObservable, ψ::DenseMPS)
    Obs.values = calObs(ψ,Obs.forest)
    Obs.forest = nothing
end

function calObs!(Obs::MPSObservable, Env::Environment)
    Obs.values = calObs(Env.layer[1],Obs.forest)
    Obs.forest = nothing
end

function calObs(ψ::DenseMPS{L,T},
    Obsf::ObserableForest) where {L,T}
    Roots = Obsf.Roots.children
    ObsDict = Dict{String,Dict}()
    for Root in Roots
        tempDict = Dict{Tuple,Number}()
        for subRoot in Root.children
            cutparent!(subRoot)
            tempDict[subRoot.Opr.name] = let 
                Env = Environment([ψ, AutomataSparseMPO(InteractionTree(subRoot),L), adjoint(ψ)])
                initialize!(Env)
                scalar(Env)
            end
        end
        ObsDict[Root.Opr.name] = tempDict
    end
    return ObsDict
end

function scalar(Env::Environment{3})
    @assert Env.center[1] == Env.center[2]
    _inproduct(action(proj1(Env.layer[2], Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]), Env.layer[3].ts[Env.center[1]])
end

function _inproduct(A::MPSTensor{3}, B::MPSTensor{3})
    return _inproduct(A,adjoint(B))
end

function _inproduct(A::MPSTensor{3}, B::AdjointMPSTensor{3})
    return @tensor A.A[1,2,3] * B.A[3,1,2]
end
