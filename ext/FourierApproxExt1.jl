module FourierApproxExt1

using PeriodicSystems
using Reexport
@reexport using PeriodicMatrices
@reexport using PeriodicMatrixEquations
using ApproxFun
using DescriptorSystems
using FastLapackInterface
using IRKGaussLegendre
using LinearAlgebra
using LineSearches
using MatrixEquations
using MatrixPencils
using Optim
using OrdinaryDiffEq
using PeriodicSchurDecompositions
using QuadGK
using SparseArrays

import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex


include("types/PeriodicStateSpace_Fourier.jl")
include("ps_Fourier.jl")
include("psconversions_Fourier.jl")
include("pslifting_Fourier.jl")
include("pstimeresp_Fourier.jl")
include("psanalysis_Fourier.jl")

end