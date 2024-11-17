function Hamiltonian(Latt::AbstractLattice, J₁::Number, J₂::Number)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = U₁Spin
    
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J₁,nothing)
            addIntr!(Root,LocalSpace.S₊S₋,pair,("S₊","S₋"),J₂,nothing)
            addIntr!(Root,LocalSpace.S₋S₊,pair,("S₋","S₊"),J₂,nothing)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    return H
end