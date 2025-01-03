module PeriodicSystems

using Reexport
@reexport using PeriodicMatrices
@reexport using PeriodicMatrixEquations
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
using Symbolics

import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, reverse,
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import DescriptorSystems: isstable, horzcat, vertcat, blockdiag, parallel, series, append, isconstant
import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex
import PeriodicMatrices: iscontinuous, isdiscrete

export PeriodicStateSpace
export ps, islti, ps_validation
export psaverage, psc2d, psmrc2d, psteval, pseval, psparallel, psseries, psappend, pshorzcat, psvertcat, psinv, psfeedback
export ps2fls, ps2frls, ps2ls, ps2spls
export pspole, pszero, isstable, psh2norm, pshanorm, pslinfnorm, pstimeresp, psstepresp, tvtimeresp, tvh2norm
export psfeedback, pssfeedback, pssofeedback
export pcpofstab_sw, pcpofstab_hr, pdpofstab_sw, pdpofstab_hr, pclqr, pclqry, pdlqr, pdlqry, pdkeg, pckeg, pdkegw, pckegw, pdlqofc, pdlqofc_sw, pclqofc_sw, pclqofc_hr

abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractLTVSystem <: AbstractDynamicalSystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPeriodicStateSpace <: AbstractLTVSystem end

include("types/PeriodicStateSpace.jl")
include("ps.jl")
include("psconversions.jl")
include("pslifting.jl")
include("pstimeresp.jl")
include("psops.jl")
include("psanalysis.jl")
include("psstab.jl")

end
