function Hamiltonian(Latt::AbstractLattice; t::Number = 1, μ::Number = 0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = U₁Fermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",-μ,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),-t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    return H
end