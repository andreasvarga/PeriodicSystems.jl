module PeriodicSystems

const withAPFUN = true

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
using Optim
#using Optimization, OptimizationOptimJL
#using StaticArrays
#using DifferentialEquations
using OrdinaryDiffEq
using IRKGaussLegendre
using Primes
withAPFUN && (using ApproxFun)
using PeriodicSchurDecompositions
#using JLD

include("SLICOTtools.jl")
using .SLICOTtools: mb03vd!, mb03vy!, mb03bd!, mb03wd!, mb03vw!, mb03kd!


import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint, opnorm, normalize, rdiv!, issymmetric, norm, tr
import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, reverse,
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import MatrixPencils: isregular, rmeval
import MatrixEquations: sylvd2!, luslv!
import DescriptorSystems: isstable, horzcat, vertcat
import Polynomials: AbstractRationalFunction, AbstractPolynomial, poles, isconstant, variable, degree, pqs
import Symbolics: derivative

export withAPFUN

export PeriodicStateSpace, pschur, pschur!, pschur1, pschur2, pgschur, pgschur!, phess, phess!, phess1, psreduc_reg, psreduc_fast, check_psim, mshift, pseig, 
       tvmeval, tpmeval, hreval, tvstm, psordschur!, psordschur1!, pgordschur!
export ts2hr, ts2pfm, tsw2pfm, ts2ffm, pfm2hr, hr2psm, psm2hr, pm2pa, ffm2hr, pmaverage, hrtrunc, hrchop
export monodromy, psceig, psceighr, psceigfr
export PeriodicArray, PeriodicMatrix, SwitchingPeriodicMatrix
export PeriodicTimeSeriesMatrix, PeriodicSwitchingMatrix, HarmonicArray, FourierFunctionMatrix, PeriodicFunctionMatrix,  PeriodicSymbolicMatrix
export isperiodic, isconstant, iscontinuous, islti, set_period
export mb03vd!, mb03vy!, mb03bd!, mb03wd!, mb03kd! 
export ps
export psaverage, psc2d, psmrc2d, psteval
export ps2fls, hr2bt, hr2btupd, phasemat, ps2frls, DiagDerOp, ps2ls
export pspole, pszero, isstable, psh2norm, pshanorm, pstimeresp, psstepresp
export pdlyap, prdlyap, pfdlyap, pslyapd, pdlyaps!, pdlyaps1!, pdlyaps2!, pdlyaps3!, dpsylv2, dpsylv2!, pslyapdkr, dpsylv2krsol!, kronset!
export prdplyap, pfdplyap, pdplyap, psplyapd
export pmshift, trace
export pclyap, pfclyap, prclyap, pgclyap, tvclyap_eval
export pcplyap, pfcplyap, prcplyap, pgcplyap, tvcplyap_eval
export pcric, prcric, pfcric, tvcric, pgcric, prdric, pfdric, tvcric_eval
export derivative, pmrand, horzcat, vertcat
#export psfeedback
#export pspofstab_sw, pspofstab_hr, plqr

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
include("psanalysis.jl")
include("pstimeresp.jl")
include("pslyap.jl")
include("psclyap.jl")
include("pscric.jl")
include("psdric.jl")
#include("psstab.jl")
#include("psops.jl")
include("pmops.jl")
include("psfutils.jl")
include("psutils.jl")
end
