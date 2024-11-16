

function Hamiltonian(Latt::AbstractLattice,J::Number,h::Number,hz::Number=0)
    H = let 
        Root = InteractionTreeNode()
        LocalSpace = TrivialSpinOneHalf
    
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.Sx,i,"Sx",h,nothing)
            addIntr!(Root,LocalSpace.Sz,i,"Sz",hz,nothing)
        end
        
        for pair in neighbor(Latt)
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J,nothing)
            addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
        end
    
        AutomataSparseMPO(InteractionTree(Root),size(Latt))
    end

    return H
end
