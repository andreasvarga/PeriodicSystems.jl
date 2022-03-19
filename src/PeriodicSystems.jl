module PeriodicSystems

#using SLICOTMath
using SLICOT_jll
#using DescriptorSystems
using LinearAlgebra
using FFTW
using MatrixEquations
using MatrixPencils
using Polynomials
using Random
using Interpolations
using Symbolics
#using StaticArrays
#using DifferentialEquations
using OrdinaryDiffEq
using DocStringExtensions
using IRKGaussLegendre

import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint, opnorm, normalize, rdiv!
import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, 
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import MatrixPencils: isregular, rmeval
import Polynomials: AbstractRationalFunction, AbstractPolynomial, poles, isconstant, variable, degree, pqs

export PeriodicDiscreteStateSpace, pschur, phess, psreduc_reg, psreduc_fast, check_psim, mshift, ts2hr, ts2pfm, pseig, tvmeval, hr2psm, hreval, tvstm
export monodromy, psceig
export PeriodicArray, PeriodicMatrix
export PeriodicTimeSeriesMatrix, HarmonicArray, PeriodicFunctionMatrix,  PeriodicSymbolicMatrix


abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractLTVSystem <: AbstractDynamicalSystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPeriodicStateSpace <: AbstractLTVSystem end

abstract type AbstractPeriodicArray{Domain} end

include("types/PeriodicMatrices.jl")
# include("ps.jl")
include("psutils.jl")
include("SLICOTtools.jl")
end
