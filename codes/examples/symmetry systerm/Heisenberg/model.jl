function Hamiltonian(Latt::AbstractLattice, J::Number)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = SU₂Spin
    
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SS,pair,("S","S"),J,nothing)
        end

        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end
    
    return H
end