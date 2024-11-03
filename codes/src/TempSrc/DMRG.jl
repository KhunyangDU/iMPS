
function DMRG2!(Env::Environment{3}, D_MPS::Int64;
    Nsweep::Int64=5, LanczosLevel::Int64=15)

    ψ = Env.layer[1]
    H = Env.layer[2]
    L = Env.L

    lsE = []

    totaltruncerror = 0
    temptruncerr = 0
    for i in 1:Nsweep

        @time "sweep $i finished, max truncation error = $(totaltruncerror)" begin
            Eg = 0
            println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                Eg,Ev = groundEig(projright2(H,Env,site),LanczosLevel)
                tl, tr, temptruncerr = tsvd(Ev; direction=:right,trunc = truncdim(D_MPS))
                pushright!(Env,map(MPSTensor, [tl, tr])...)
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
            println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                Eg,Ev = groundEig(projleft2(H,Env,site),LanczosLevel)
                tl, tr, temptruncerr = tsvd(Ev; direction=:left,trunc = truncdim(D_MPS))
                pushleft!(Env,map(MPSTensor, [tl, tr])...)
                totaltruncerror = max(totaltruncerror,temptruncerr)
            end
            push!(lsE, Eg)
        end
        
        GC.gc()
    end
    
    return lsE
end

function pushright!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site:site+1] = [tl,tr]
    Env.layer[3].ts[site:site+1] = adjoint(Env.layer[1].ts[site:site+1])
    pushright!(Env)
end

function pushleft!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site-1:site] = [tl, tr]
    Env.layer[3].ts[site-1:site] = adjoint(Env.layer[1].ts[site-1:site])
    pushleft!(Env)
end

function DMRG2!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    lsE = DMRG2!(Env, D_MPS;kwargs...)
    return Env.layer[1], lsE
end
