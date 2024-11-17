
function Hamiltonian(Latt::AbstractLattice;
    J::Number=1,hx::Number=0,hy::Number=0,hz::Number = 0,
    returntree::Bool=false)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sx,i,"Sx",hx,nothing)
        addIntr!(Root,LocalSpace.Sy,i,"Sy",hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",hz,nothing)
    end
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
    end

    if returntree
        return InteractionTree(Root)
    else
        return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    end
    
end

