module PeriodicSystems

#using SLICOTMath
using SLICOT_jll
using DescriptorSystems
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
using IRKGaussLegendre
using Primes
using ApproxFun

include("SLICOTtools.jl")
using .SLICOTtools: mb03vd!, mb03vy!, mb03bd!, mb03wd!


import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint, opnorm, normalize, rdiv!
import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, 
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import MatrixPencils: isregular, rmeval
import Polynomials: AbstractRationalFunction, AbstractPolynomial, poles, isconstant, variable, degree, pqs

export PeriodicStateSpace, pschur, phess, pschurw, psreduc_reg, psreduc_fast, check_psim, mshift, pseig, tvmeval, hreval, tvstm
export ts2hr, ts2pfm, ts2ffm, pfm2hr, hr2psm, psm2hr, pm2pa, ffm2hr, pmaverage
export monodromy, psceig
export PeriodicArray, PeriodicMatrix
export PeriodicTimeSeriesMatrix, HarmonicArray, FourierFunctionMatrix, PeriodicFunctionMatrix,  PeriodicSymbolicMatrix
export isperiodic, isconstant, iscontinuous, islti, set_period
export mb03vd!, mb03vy!, mb03bd!, mb03wd!
export ps
export psaverage, psc2d, psmrc2d
export ps2fls, hr2bt, hr2btupd, phasemat, ps2frls, DiagDerOp

abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractLTVSystem <: AbstractDynamicalSystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPeriodicStateSpace <: AbstractLTVSystem end

abstract type AbstractPeriodicArray{Domain,T} end

include("types/PeriodicMatrices.jl")
include("types/PeriodicStateSpace.jl")
include("ps.jl")
include("conversions.jl")
include("pslifting.jl")
include("psutils.jl")
end
