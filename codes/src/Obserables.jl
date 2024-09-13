
function Quant1(ψ::Vector,Q::AbstractTensorMap{ComplexSpace,1,1},D_MPS::Int64)

    L = length(ψ)

    ψcp = Vector{AbstractTensorMap}(undef,L)
    lsQi = Vector{Float64}(undef,L)
    
    ψcp[1] = deepcopy(ψ[1])
    for i in 1:L
#=         effQ = EffHam(Q,InitialLeftEnv(),InitialRightEnv())
        Qi = @tensor ψcp[i][1,5,2]*effQ[1,5,2,3,6,4]*ψcp[i]'[3,6,4] =#

        Qi = @tensor ψcp[i][1,3,2]*Q[3,4]*ψcp[i]'[1,4,2]

        lsQi[i] = ApproxReal(Qi)

        if i < L
            ψcp[i:i+1],truncerr = RightMove(ψ[i+1],ψcp[i],D_MPS)
        end
        
    end

    return lsQi
end

function QuantUniv(ψ::Vector,Q::Vector)
    effQ = EffHam(Q[1],LeftEnv(ψ,Q,1),RightEnv(ψ,Q,ψ,1))
    q = @tensor ψ[1][1,5,2]*effQ[1,5,2,3,6,4]*ψ[1]'[3,6,4]
    return ApproxReal(q)
end


function GreenFuncRet(ψ::Vector,H::Vector,E0::Number,
    CK::Vector,CKdagg::Vector,lsE::Vector,D_MPS::Int64;
    τ::Number = 1/abs(E0)/5,TruncErr::Number = 1e-5,MaxIter::Int64=30)
    # calculate the G^{ret}(k,ω) with given cₖ,cₖ⁺
    # how to choose evolve step τ?
    # spectrum function is S(k,ω) = - Im G^{ret}(k,ω) / π

    Gk = Vector{ComplexF64}(undef,length(lsE))

    ψ₊E = VariContract(CKdagg,ψ,D_MPS)
    ψ₋E = VariContract(CK,ψ,D_MPS)

    lsGt₊,lst₊ = GreenFuncTDVP2(ψ₊E,H,τ,TruncErr,MaxIter,D_MPS)
    lsGt₋,lst₋ = GreenFuncTDVP2(ψ₋E,H,-τ,TruncErr,MaxIter,D_MPS)
    #@show lsGt₊ .+ lsGt₋
    #Gk = [sum(@. (-1im)*lsGt₊*exp(1im*(E+E0)*lst₊))*τ+sum(@. (-1im)*lsGt₋*exp(1im*(E-E0)*lst₊))*τ for E in lsE]

    Gk = [sum(@. (-1im)*lsGt₊*exp(1im*(E+E0)*lst₊))*τ+sum(@. (-1im)*lsGt₋*exp(-1im*(E-E0)*lst₋))*τ for E in lsE]
    #Gk = [sum(@. (-1im)*lsGt₋*exp(1im*(E-E0)*lst₋))*τ for E in lsE]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    return Gk
end




