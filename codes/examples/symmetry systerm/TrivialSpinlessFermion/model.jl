function Hamiltonian(Latt::AbstractLattice,t::Number,μ::Number)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinlessFermion
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.n,i,"n",-μ,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),t,LocalSpace.Z)
            addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end

    return H
end