using Pkg

Pkg.activate(".")

Pkg.add("TensorKit")
Pkg.add("JLD2")
Pkg.add("LinearAlgebra")
Pkg.resolve()
Pkg.gc()

