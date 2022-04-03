# Base.promote_rule(PeriodicFunctionMatrix, PeriodicSymbolicMatrix) = PeriodicFunctionMatrix{:c,T}
# Base.promote_rule(PeriodicFunctionMatrix, HarmonicArray) = PeriodicFunctionMatrix
function promote_period(PM1,args...; ndigits = 4)
    period = PM1.period
    nlim = 2^ndigits
    isconst = isconstant(PM1)
    for a in args
        isconstant(a) && continue
        isconst && (isconst = false; period = a.period; continue)
        peri = a.period
        r = rationalize(period/peri)
        num = numerator(r)
        den = denominator(r)
        (num <= nlim && den <= nlim) || error("incommensurate periods")
        period = period*den
    end
    return period
end
function promote_Ts(PM1,args...)
    Ts = PM1.Ts
    isconst = isconstant(PM1)
    for a in args
        isconstant(a) && continue
        isconst && (isconst = false; Ts = a.Ts; continue)
        Ts ≈ a.Ts || error("incompatible sampling times")
    end
    return Ts
end

struct PeriodicStateSpace{PM} <: AbstractPeriodicStateSpace
    A::PM
    B::PM
    C::PM
    D::PM
    period::Float64
end
"""
    PeriodicStateSpace(A::PM, B::PM, C::PM, D::PM) -> psys::PeriodicStateSpace{PM}

Construct a `PeriodicStateSpace` object from a quadruple of periodic matrix objects. 

The periodic matrix objects `A`, `B`, `C`, `D` specifies the periodic matrices
`A(t)`, `B(t)`, `C(t)` and `D(t)` of a linear periodic time-varying state space model
in the continuous-time form

     dx(t)/dt = A(t)x(t) + B(t)u(t) ,
     y(t)     = C(t)x(t) + D(t)u(t) ,

or in the discrete-time form

     x(t+1)  = A(t)x(t) + B(t)u(t) ,
     y(t)    = C(t)x(t) + D(t)u(t) , 

where `x(t)`, `u(t)` and `y(t)` are the system state vector, 
system input vector and system output vector, respectively, 
and `t` is the continuous or discrete time variable. 
The system matrices satisfy `A(t) = A(t+T₁)`, `B(t) = B(t+T₂)`, `C(t) = C(t+T₃)`, `D(t) = D(t+T₄)`,  
i.e., are periodic with periods `T₁`, `T₂`, `T₃` and `T₄`, respectively. 
The different periods must be commensurate (i.e., their ratios must be rational numbers with
numerators and denominators up to at most 4 decimal digits). 
All periodic matrix objects must have the same type `PM`, where
`PM` stays for one of the supported periodic matrix types, i.e., 
[`PeriodicMatrix`](@ref), [`PeriodicArray`](@ref), [`PeriodicFunctionMatrix`](@ref), [`PeriodicSymbolicMatrix`](@ref),
[`HarmonicArray`](@ref) or [`PeriodicTimeSeriesMatrix`](@ref). 
"""
function PeriodicStateSpace(A::PFM, B::PFM, C::PFM, D::PFM) where {T,PFM <: PeriodicFunctionMatrix{:c,T}}
    period = ps_validation(A, B, C, D)
    PeriodicStateSpace{PFM}(period == A.period ? A : PeriodicFunctionMatrix(A,period), period == B.period ? B : PeriodicFunctionMatrix(B,period), 
                            period == C.period ? C : PeriodicFunctionMatrix(C,period), period == D.period ? D : PeriodicFunctionMatrix(D,period), 
                            Float64(period))
end
function PeriodicStateSpace(A::PSM, B::PSM, C::PSM, D::PSM) where {T,PSM <: PeriodicSymbolicMatrix{:c,T}}
    period = ps_validation(A, B, C, D)
    PeriodicStateSpace{PSM}(period == A.period ? A : PeriodicSymbolicMatrix(A,period), period == B.period ? B : PeriodicSymbolicMatrix(B,period), 
                            period == C.period ? C : PeriodicSymbolicMatrix(C,period), period == D.period ? D : PeriodicSymbolicMatrix(D,period), 
                            Float64(period))
end
function PeriodicStateSpace(A::PHR1, B::PHR2, C::PHR3, D::PHR4) where {PHR1 <: HarmonicArray, PHR2 <: HarmonicArray, PHR3 <: HarmonicArray, PHR4 <: HarmonicArray}
    period = ps_validation(A, B, C, D)
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    PeriodicStateSpace{HarmonicArray{:c,T}}((period == A.period && T == eltype(A)) ? A : HarmonicArray{:c,T}(A,period), 
                                            (period == B.period && T == eltype(B)) ? B : HarmonicArray{:c,T}(B,period), 
                                            (period == C.period && T == eltype(C)) ? C : HarmonicArray{:c,T}(C,period), 
                                            (period == D.period && T == eltype(D)) ? D : HarmonicArray{:c,T}(D,period), 
                                            Float64(period))
end
function PeriodicStateSpace(A::PTSM1, B::PTSM2, C::PTSM3, D::PTSM4) where {PTSM1 <: PeriodicTimeSeriesMatrix, PTSM2 <: PeriodicTimeSeriesMatrix, PTSM3 <: PeriodicTimeSeriesMatrix, PTSM4 <: PeriodicTimeSeriesMatrix}
    period = ps_validation(A, B, C, D)
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    PeriodicStateSpace{PeriodicTimeSeriesMatrix{:c,T}}((period == A.period && T == eltype(A)) ? A : PeriodicTimeSeriesMatrix{:c,T}(A,period), 
                                            (period == B.period && T == eltype(B)) ? B : PeriodicTimeSeriesMatrix{:c,T}(B,period), 
                                            (period == C.period && T == eltype(C)) ? C : PeriodicTimeSeriesMatrix{:c,T}(C,period), 
                                            (period == D.period && T == eltype(D)) ? D : PeriodicTimeSeriesMatrix{:c,T}(D,period), 
                                            Float64(period))
end

function ps_validation(A::PM1, B::PM2, C::PM3, D::PM4) where {T1, T2, T3, T4, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}, PM4 <: AbstractPeriodicArray{:c,T4}}
    nx = size(A,1)
    nx == size(A,2) || DimensionMismatch("matrix A(t) is not square")
    (ny, nu) = size(D)
    # validate dimensions
    size(B,1) == nx ||  DimensionMismatch("B(t) must have the same row size as A(t)")
    size(C,2) == nx ||  DimensionMismatch("C(t) must have the same column size as A(t)")
    nu == size(B,2) ||  DimensionMismatch("D(t) must have the same column size as B(t)")
    ny == size(C,1) ||  DimensionMismatch("D(t) must have the same row size as C(t)")
    # validate sampling time
    period = promote_period(A, B, C, D)
    return period
end
function PeriodicStateSpace(A::PM1, B::PM2, C::PM3, D::PM4) where {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix, PM4 <: PeriodicMatrix}
    period = ps_validation(A, B, C, D)
    promote_Ts(A,B,C,D)
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    PeriodicStateSpace{PeriodicMatrix{:d,T}}((period == A.period && T == eltype(A)) ? A : PeriodicMatrix{:d,T}(A,period), 
                                             (period == B.period && T == eltype(B)) ? B : PeriodicMatrix{:d,T}(B,period), 
                                             (period == C.period && T == eltype(C)) ? C : PeriodicMatrix{:d,T}(C,period), 
                                             (period == D.period && T == eltype(D)) ? D : PeriodicMatrix{:d,T}(D,period), 
                                             Float64(period))
end

function ps_validation(A::PM1, B::PM2, C::PM3, D::PM4) where {T1, T2, T3, T4, PM1 <: PeriodicMatrix{:d,T1}, PM2 <: PeriodicMatrix{:d,T2}, PM3 <: PeriodicMatrix{:d,T3}, PM4 <: PeriodicMatrix{:d,T4}}
    period = promote_period(A, B, C, D)
    max(length(A),length(B),length(C),length(D)) == 0 && (return 0, 0, Int[], period)

    # Validate dimensions
    ny = size(D,1)
    nu = size(D,2)
    any(ny .!= ny[1]) && DimensionMismatch("all matrices D[i] must have the same row size")
    any(nu .!= nu[1]) && DimensionMismatch("all matrices D[i] must have the same column size")
    
    ndx, nx = size(A)
    if all(ndx .== ndx[1]) && all(nx .== nx[1]) 
       # constant dimensions of A, B, C, D
       ndx[1] == nx[1] || DimensionMismatch("A[i] must be square matrices")
       any(size(B, 1) .!= ndx[1]) && DimensionMismatch("all matrices A[i] and B[i] must have the same row size")
       any(size(C, 2) .!= nx[1]) &&  DimensionMismatch("all matrices A[i] and C[i] must have the same column size")
       any(size(C, 1) .!= ny[1]) && DimensionMismatch("all matrices C[i] and D[i] must have the same row size")
       any(size(B, 2) .!= nu[1]) && DimensionMismatch("all matrices B[i] and D[i] must have the same column size")
    else     
       N = A.nperiod
       (N != B.nperiod || N != C.nperiod) && DimensionMismatch("the number of component matrices of A, B and C must be the same ")
       for i = 1:N
           i == N && ndx[i] != nx[1] && DimensionMismatch("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
           i != N && ndx[i] != nx[i+1] && DimensionMismatch("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
       end
       any(size(B, 1) .!= ndx) && DimensionMismatch("all matrices A[i] and B[i] must have the same row size")
       any(size(C, 2) .!= nx) &&  DimensionMismatch("all matrices A[i] and C[i] must have the same column size")
       any(size(C, 1) .!= ny[1]) && DimensionMismatch("all matrices C[i] and D[i] must have the same row size")
       any(size(B, 2) .!= nu[1]) && DimensionMismatch("all matrices B[i] and D[i] must have the same column size")
    end
    return period
end
function PeriodicStateSpace(A::PA1, B::PA2, C::PA3, D::PA4) where {PA1 <: PeriodicArray, PA2 <: PeriodicArray, PA3 <: PeriodicArray, PA4 <: PeriodicArray}
    period = ps_validation(A, B, C, D)
    promote_Ts(A,B,C,D)
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    PeriodicStateSpace{PeriodicArray{:d,T}}((period == A.period && T == eltype(A)) ? A : PeriodicArray{:d,T}(A,period), 
                                            (period == B.period && T == eltype(B)) ? B : PeriodicArray{:d,T}(B,period),  
                                            (period == C.period && T == eltype(C)) ? C : PeriodicArray{:d,T}(C,period),  
                                            (period == D.period && T == eltype(D)) ? D : PeriodicArray{:d,T}(D,period),  
                                            Float64(period))
end
function ps_validation(A::PA1, B::PA2, C::PA3, D::PA4) where {T1, T2, T3, T4, PA1 <: PeriodicArray{:d,T1}, PA2 <: PeriodicArray{:d,T2}, PA3 <: PeriodicArray{:d,T3}, PA4 <: PeriodicArray{:d,T4}}
    period = promote_period(A, B, C, D)
    max(length(A),length(B),length(C),length(D)) == 0 && (return 0, 0, Int[], period)

    # Validate dimensions
    ny = size(D,1)
    nu = size(D,2)
    nx = size(A,1)
    nx == size(A,2) || DimensionMismatch("matrix A(t) is not square")
    
    any(size(B, 1) .!= nx) && DimensionMismatch("A and B must have the same row size")
    any(size(C, 2) .!= nx) &&  DimensionMismatch("A and C must have the same column size")
    any(size(C, 1) .!= ny) && DimensionMismatch("C and D must have the same row size")
    any(size(B, 2) .!= nu) && DimensionMismatch("B and D must have the same column size")
 
    return period
end

#  conversions
function Base.convert(::Type{PeriodicStateSpace{PM}}, psys::PeriodicStateSpace) where {PM <: AbstractPeriodicArray}
    PeriodicStateSpace(convert(PM,psys.A), convert(PM,psys.B), convert(PM,psys.C), convert(PM,psys.D))
end
# properties
Base.size(sys::PeriodicStateSpace) = maximum.(size(sys.D))
function Base.size(sys::PeriodicStateSpace, d::Integer)
    return d <= 2 ? size(sys)[d] : 1
end
Base.length(sys::PeriodicStateSpace) = 1
Base.eltype(sys::PeriodicStateSpace) = eltype(sys.A)

function Base.getproperty(sys::PeriodicStateSpace, d::Symbol)  
    if d === :nx
        return size(getfield(sys, :A), 2)
    elseif d === :ny
        return size(getfield(sys, :C), 1)
    elseif d === :nu
        return size(getfield(sys, :B), 2)
    else
        getfield(sys, d)
    end
end
Base.propertynames(sys::PeriodicStateSpace) =
    (:nx, :ny, :nu, fieldnames(typeof(sys))...)


# display sys
Base.print(io::IO, sys::PeriodicStateSpace) = show(io, sys)
Base.show(io::IO, sys::PeriodicStateSpace{PM}) where {PM <: Union{PeriodicMatrix,PeriodicArray}} = show(io, MIME("text/plain"), sys)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::PeriodicStateSpace{<:PeriodicMatrix})
    summary(io, sys); println(io)
    n = sys.nx 
    N = sys.A.dperiod
    p, m = size(sys)
    period = sys.period
    Ts = sys.A.Ts
    if any(n .> 0)
       dperiod, nperiod = sys.A.dperiod, sys.A.nperiod
       subperiod = period/nperiod
       println(io, "\nState matrix A: subperiod: $subperiod    #subperiods: $nperiod ")
       println(io, "time values: t[1:$dperiod]")
       for i = 1:dperiod
           println("t[$i] = $(i*Ts)")
          display(mime, sys.A.M[i])
       end
       if m > 0 
          dperiod, nperiod = sys.B.dperiod, sys.B.nperiod
          subperiod = period/nperiod
          println(io, "\n\nInput matrix B: subperiod: $subperiod    #subperiods: $nperiod ") 
          println(io, "time values: t[1:$dperiod]")
          for i = 1:dperiod
              println("t[$i] = $(i*Ts)")
              display(mime, sys.B.M[i])
          end
       else
          println(io, "\n\nEmpty input matrix B.")
       end
       if p > 0 
          dperiod, nperiod = sys.C.dperiod, sys.C.nperiod
          subperiod = period/nperiod
          println(io, "\n\nOutput matrix C: subperiod: $subperiod    #subperiods: $nperiod ")
          println(io, "time values: t[1:$dperiod]")
          for i = 1:dperiod
              println("t[$i] = $(i*Ts)")
             display(mime, sys.C.M[i])
          end
       else
          println(io, "\n\nEmpty output matrix C.") 
       end
       if m > 0 && p > 0
          dperiod, nperiod = sys.D.dperiod, sys.D.nperiod
          subperiod = period/nperiod
          println(io, "\n\nFeedthrough matrix D: subperiod: $subperiod    #subperiods: $nperiod ") 
          println(io, "time values: t[1:$dperiod]")
          for i = 1:dperiod
              println("t[$i] = $(i*Ts)")
              display(mime, sys.D.M[i])
          end
       else
           println(io, "\n\nEmpty feedthrough matrix D.") 
       end
       println(io, "\n\nPeriod:      $(sys.period) second(s).")
       println(io,     "Sample time: $Ts second(s).")
       println(io, "Periodic discrete-time state-space model.") 
    elseif m > 0 && p > 0
       dperiod, nperiod = sys.D.dperiod, sys.D.nperiod
       subperiod = period/nperiod
       println(io, "\n\nFeedthrough matrix D: subperiod: $subperiod    #subperiods: $nperiod ") 
       println(io, "time values: t[1:$dperiod]")
       for i = 1:dperiod
           println("t[$i] = $(i*Ts)")
           display(mime, sys.D.M[i])
       end
       println(io, "\n\nPeriod:      $(sys.period) second(s).")
       println(io,     "Sample time: $Ts second(s).")
       println(io, "Time-varying gains.") 
    else
       println(io, "\nEmpty Periodic discrete-time state-space model.")
    end
end
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::PeriodicStateSpace{<:PeriodicArray})
    summary(io, sys); println(io)
    n = sys.nx 
    N = sys.A.dperiod
    p, m = size(sys)
    period = sys.period
    Ts = sys.A.Ts
    if any(n .> 0)
        dperiod, nperiod = sys.A.dperiod, sys.A.nperiod
        subperiod = period/nperiod
        println(io, "\nState matrix A: subperiod: $subperiod    #subperiods: $nperiod ")
        println(io, "time values: t[1:$dperiod]")
        for i = 1:dperiod
            println("t[$i] = $(i*Ts)")
            display(mime, sys.A.M[:,:,i])
        end
        if m > 0 
           dperiod, nperiod = sys.B.dperiod, sys.B.nperiod
           subperiod = period/nperiod
           println(io, "\n\nInput matrix B: subperiod: $subperiod    #subperiods: $nperiod ") 
           println(io, "time values: t[1:$dperiod]")
           for i = 1:dperiod
               println("t[$i] = $(i*Ts)")
               display(mime, sys.B.M[:,:,i])
           end
        else
           println(io, "\n\nEmpty input matrix B.")
        end
        if p > 0 
           dperiod, nperiod = sys.C.dperiod, sys.C.nperiod
           subperiod = period/nperiod
           println(io, "\n\nOutput matrix C: subperiod: $subperiod    #subperiods: $nperiod ")
           println(io, "time values: t[1:$dperiod]")
           for i = 1:dperiod
               println("t[$i] = $(i*Ts)")
               display(mime, sys.C.M[:,:,i])
           end
        else
           println(io, "\n\nEmpty output matrix C.") 
        end
        if m > 0 && p > 0
           dperiod, nperiod = sys.D.dperiod, sys.D.nperiod
           subperiod = period/nperiod
           println(io, "\n\nFeedthrough matrix D: subperiod: $subperiod    #subperiods: $nperiod ") 
           println(io, "time values: t[1:$dperiod]")
           for i = 1:dperiod
               println("t[$i] = $(i*Ts)")
               display(mime, sys.D.M[:,:,i])
           end
        else
           println(io, "\n\nEmpty feedthrough matrix D.") 
        end
        println(io, "\n\nPeriod:      $(sys.period) second(s).")
        println(io,     "Sample time: $Ts second(s).")
        println(io, "Periodic discrete-time state-space model.") 
    elseif m > 0 && p > 0
        dperiod, nperiod = sys.D.dperiod, sys.D.nperiod
        subperiod = period/nperiod
        println(io, "\n\nFeedthrough matrix D: subperiod: $subperiod    #subperiods: $nperiod ") 
        println(io, "time values: t[1:$dperiod]")
        for i = 1:dperiod
            println("t[$i] = $(i*Ts)")
            display(mime, sys.D.M[:,:,i])
        end
        println(io, "\n\nPeriod:      $(sys.period) second(s).")
        println(io,     "Sample time: $Ts second(s).")
        println(io, "Time-varying gains.") 
    else
        println(io, "\nEmpty Periodic discrete-time state-space model.")
    end
end

 
# """ 
#     PeriodicDiscreteStateSpace{T}(A::Vector{Matrix{T}}, E::Vector{Union{Matrix{T},UniformScaling}}, 
#                             B::Vector{Matrix{T}}, C::Vector{Matrix{T}}, D::Vector{Matrix{T}},  
#                             Ts::Real) where T <: Number 

# Construct a periodic discrete-time descriptor state-space model from a quintuple of 
# `N`-dimensional vectors of matrices `(A,E,B,C,D)` and a sampling time `Ts`. 
# `N` is the period of the system.

# If `SYS::PeriodicDiscreteStateSpace{T}` is a periodic descriptor system model object 
# defined by the 5-tuple `(A,E,B,C,D)`, then:

# `SYS.A[i]` is the `ndx[i] × nx[i]` state matrix `A[i]` with elements of type `T`. 
# The dimensions must satisfy `ndx[i] = nx[i+1]`, for `i = 1, ..., N-1` and `ndx[N] = nx[1]`.

# `SYS.E[i]` is the `ndx[i] × ndx[i]` descriptor matrix `E[i]` with elements of type `T`.
#  For a standard periodic discrete-time system, for all `i` `SYS.E[i] = I`, the `UniformScaling` of type `Bool`. 

# `SYS.B[i]` is the `ndx[i] × nu[i]` system input matrix `B[i]` with elements of type `T`. 
# All input dimensions must be the same.

# `SYS.C[i]` is the `nx[i] × ny[i]` system output matrix `C[i]` with elements of type `T`. 

# `SYS.D` is the `ny × nu` system feedthrough matrix `D` with elements of type `T`. 
# All output dimensions must be the same.

# `SYS.Ts` is the real sampling time `Ts`, where `Ts > 0` or `Ts = -1`, which 
# indicates a discrete-time system with an unspecified sampling time. 

# The vector of dimensions `nx`, ndx, `ny`, `nu` can be obtained as `SYS.nx`, `SYS.ndx`, `SYS.ny` and `SYS.nu`, respectively, 
# the period N can be obtained as  `SYS.period`.
# """
# struct PeriodicDiscreteDescriptorStateSpace{T} <: AbstractPeriodicStateSpace 
#     A::Vector{Matrix{T}}
#     E::Union{Vector{Matrix{T}},UniformScaling{Bool}}
#     B::Vector{Matrix{T}}
#     C::Vector{Matrix{T}}
#     D::Vector{Matrix{T}}
#     Ts::Float64
#     function PeriodicDiscreteDescriptorStateSpace{T}(A::Vector{Matrix{T}}, E::Union{Vector{Matrix{T}},UniformScaling{Bool}}, 
#                 B::Vector{Matrix{T}}, C::Vector{Matrix{T}}, D::Vector{Matrix{T}},  Ts::Real) where T
#         ps_validation(A, E, B, C, D, Ts)
#         new{T}(A, E, B, C, D, Float64(Ts))
#     end
# end
# function ps_validation(A::Vector{Matrix{T}}, E::Union{Vector{Matrix{T}},UniformScaling{Bool}},  
#          B::Vector{Matrix{T}}, C::Vector{Matrix{T}}, D::Vector{Matrix{T}},  Ts::Real) where T
#     N = length(A)
#     desc = (E != I)
#     ((desc && (N != length(E))) || N != length(B) || N != length(C) || N != length(D)) && 
#         error("all vectors of state-space matrices must have the same length") 
#     # Validate sampling time
#     Ts > 0 || Ts == -1 || error("Ts must be either a positive number or -1 (unspecified)")
#     N == 0 && (return N)

#     # Validate dimensions
#     ny = size.(D,1)
#     nu = size.(D,2)
#     # any(ny .!= ny[1]) && error("all matrices D[i] must have the same row size")
#     # any(nu .!= nu[1]) && error("all matrices D[i] must have the same column size")
#     ndx = size.(A,1)
#     nx = size.(A,2)
    
#     for i = 1:N
#         if desc
#            ndx[i] == LinearAlgebra.checksquare(E[i]) || error("all matrices A[i] and E[i] must have the same row size")
#         end
#         i == N && ndx[i] != nx[1] && error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
#         i != N && ndx[i] != nx[i+1] && error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
#     end
#     any(size.(B, 1) .!= ndx) && error("all matrices A[i] and B[i] must have the same row size")
#     any(size.(C, 2) .!= nx) &&  error("all matrices A[i] and C[i] must have the same column size")
#     any(size.(C, 1) .!= ny) && error("all matrices C[i] and D[i] must have the same row size")
#     any(size.(B, 2) .!= nu) && error("all matrices B[i] and D[i] must have the same column size")
 
#     return N
# end

# function Base.getproperty(sys::PeriodicDiscreteDescriptorStateSpace, d::Symbol)  
#     if d === :nx
#         return size.(sys.A,2)
#     elseif d === :ndx
#         return size.(sys.A,1)
#     elseif d === :ny
#         return size.(sys.C,1)
#     elseif d === :nu
#         return size.(sys.B,2)
#     elseif d === :period
#         return length(sys.A)
#     else
#         getfield(sys, d)
#     end
# end
# Base.propertynames(sys::PeriodicDiscreteDescriptorStateSpace) =
#     (:nx, :ndx, :ny, :nu, :period, fieldnames(typeof(sys))...)
# """
#     size(sys) -> (p,m)
#     size(sys,1) -> p
#     size(sys,2) -> m

# Return the number of outputs `p` and the number of inputs `m` of a descriptor system `sys`.
# """
# function Base.size(sys::PeriodicDiscreteDescriptorStateSpace)
#     return sys.period == 0 ? (0,0) : size(sys.D[1])
# end
# function Base.size(sys::PeriodicDiscreteDescriptorStateSpace, d::Integer)
#     return d <= 2 ? size(sys)[d] : 1
# end
# function Base.length(sys::PeriodicDiscreteDescriptorStateSpace)
#     return 1
# end
# function Base.eltype(sys::PeriodicDiscreteDescriptorStateSpace)
#     return sys.period == 0 ? Float64 : eltype(sys.A[1])
# end

# # display sys
# Base.print(io::IO, sys::PeriodicDiscreteDescriptorStateSpace) = show(io, sys)
# Base.show(io::IO, sys::PeriodicDiscreteDescriptorStateSpace) = show(io, MIME("text/plain"), sys)

# function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::PeriodicDiscreteDescriptorStateSpace)
#     summary(io, sys); println(io)
#     n = size.(sys.A,1) 
#     N = sys.period
#     p, m = size(sys)
#     desc = (sys.E != I)
#     if any(n .> 0)
#        println(io, "\nState matrix A:")
#        for i = 1:N
#            println("@time: $i")
#            display(mime, sys.A[i])
#        end
#        if desc 
#           println(io, "\n\nDescriptor matrix E:")
#           for i = 1:N
#               println("@time: $i")
#               display(mime, sys.E[i])
#           end
#        end
#        if m > 0 
#           println(io, "\n\nInput matrix B:") 
#           for i = 1:N
#               println("@time: $i")
#               display(mime, sys.B[i])
#           end
#        else
#           println(io, "\n\nEmpty input matrix B.")
#        end
#        if p > 0 
#           println(io, "\n\nOutput matrix C:")
#           for i = 1:N
#             println("@time: $i")
#             display(mime, sys.C[i])
#           end
#        else
#           println(io, "\n\nEmpty output matrix C.") 
#        end
#        if m > 0 && p > 0
#           println(io, "\n\nFeedthrough matrix D:") 
#           for i = 1:N
#               println("@time: $i")
#               display(mime, sys.D[i])
#            end
#        else
#            println(io, "\n\nEmpty feedthrough matrix D.") 
#        end
#        sys.Ts < 0 && println(io, "\n\nSample time: unspecified.")
#        sys.Ts > 0 && println(io, "\n\nSample time: $(sys.Ts) second(s).")
#        desc ? println(io, "Periodic discrete-time descriptor state-space model.") :
#               println(io, "Periodic discrete-time state-space model.") 
#     elseif m > 0 && p > 0
#        println(io, "\nFeedthrough matrix D:")
#        show(io, mime, sys.D)
#        println(io, "\n\nStatic gains.") 
#     else
#        println(io, "\nEmpty Periodic discrete-time tate-space model.")
#     end
# end


# function dss(psys::PeriodicDiscreteDescriptorStateSpace{T}; kstart::Int = 1, compacted::Bool = false, 
#              fast::Bool = true, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
#              rtol::Real = eps(real(float(one(T))))*iszero(max(atol1,atol2))) where T
#     indp(j,n) = mod(j-1,n)+1
#     desc = (psys.E != I)
#     K = psys.period
#     Ts = psys.Ts 
#     LTs = Ts > 0 ? Ts*K : -1
#     #[ap,bp,cp,dp,ep] = pdsdata(psys);
#     # generate the stacked lifted system of (Grasselli and Longhi, 1991)
#     kind = indp.(kstart:kstart+K-1,K)
#     ni = psys.nx[kind]
#     mui = psys.ndx[kind]
#     m = psys.nu[kind]
#     n = sum(ni)
#     mukm1 = sum(view(mui,1:K-1))
#     nukm1 = sum(view(ni,1:K-1))
#     KK = kind[K]
#     kindm1 = view(kind,1:K-1)
#     A = [zeros(T, mui[K], nukm1) psys.A[KK]; blockdiag(view(psys.A,kindm1)...) zeros(T, mukm1, ni[K])] +
#         [zeros(T, mui[K], n); zeros(T, mukm1, ni[1]) desc ? -blockdiag(view(psys.E,kindm1)...) : -I]
#     E = [desc ? psys.E[KK] : I zeros(T, mui[K], n-ni[1]); zeros(T, mukm1, n)]
#     B = [ zeros(T, mui[K], sum(view(m,1:K-1))) psys.B[KK]; blockdiag(view(psys.B,kindm1)...) zeros(T, mukm1, m[K])]
#     C = blockdiag(view(psys.C,kind)...) 
#     D = blockdiag(view(psys.D,kind)...) 
#     if compacted 
#        A, E, B, C, D = lsminreal(A, E, B, C, D; fast, atol1, atol2, rtol, contr = false, obs = false, noseig = true) 
#     end
#     return dss(A, E, B, C, D; Ts = LTs)
# end

# function psreduc_fast(S::Vector{Matrix{T1}}, T::Vector{Matrix{T1}}; atol::Real = 0) where T1
#     # PSREDUC_FAST  Finds a matrix pair having the same finite and infinite 
#     #               eigenvalues as a given periodic pair.
#     #
#     # [A,E] = PSREDUC_FAST(S,T,tol) computes a matrix pair (A,E) having 
#     # the same finite and infinite eigenvalues as the periodic pair (S,T).
#     #
    
#     #   A. Varga 30-3-2004. 
#     #   Revised: .
#     #
#     #   Reference:
#     #   A. Varga & P. Van Dooren
#     #      Computing the zeros of periodic descriptor systems.
#     #      Systems and Control Letters, vol. 50, 2003.
    
#     K = length(S) 
#     if K == 1
#        return S[1], T[1] 
#     else
#        m = size.(S,1)
#        n = size.(S,2)
#        if sum(m) > sum(n) 
#            # build the dual pair
#         #    [S,T]=celltranspose(S,T); 
#         #    [S{1:K}] = deal(S{K:-1:1});
#         #    [T{1:K-1}] = deal(T{K-1:-1:1});
#         #    m = size.(S,1)
#         #    n = size.(S,2)
#        end 
    
#        si = S[1];  ti = -T[1];
#        tolr = atol
#        for i = 1:K-2
#            F = qr([ ti; S[i+1] ], ColumnNorm()) 
#            println("F = $(size(F.R))")
#            nr = minimum(size(F.R))
#            # compute rank of r 
#            ss = abs.(diag(F.R[1:nr,1:nr]))
#            atol == 0 && ( tolr = (m[i]+m[i+1]) * maximum(ss) * eps())
#            rankr = count(ss .> tolr)
    
#            si = F.Q'*[si; zeros(T1,m[i+1],n[1])]; si=si[rankr+1:end,:]
#            ti = F.Q'*[ zeros(T1,size(ti,1),n[i+2]); -T[i+1]]; ti = ti[rankr+1:end,:]
#        end
#        a = [ zeros(T1,m[K],n[1]) S[K]; si ti] 
#        e = [ T[K] zeros(T1,m[K],n[K]); zeros(size(si)...) zeros(size(ti)...)] 
#        return a, e
#     end
# end





# function Base.getindex(sys::DST, inds...) where DST <: DescriptorStateSpace
#     size(inds, 1) != 2 &&
#         error("Must specify 2 indices to index descriptor state-space models")
#     rows, cols = index2range(inds...) 
#     return DescriptorStateSpace{eltype(sys)}(copy(sys.A), copy(sys.E), sys.B[:, cols], sys.C[rows, :], sys.D[rows, cols], sys.Ts)
# end
# index2range(ind1, ind2) = (index2range(ind1), index2range(ind2))
# index2range(ind::T) where {T<:Number} = ind:ind
# index2range(ind::T) where {T<:AbstractArray} = ind
# index2range(ind::Colon) = ind
# function Base.lastindex(sys::DST, dim::Int) where DST <: DescriptorStateSpace
#     lastindex(sys.D,dim)
# end
# # Basic Operations
# function ==(sys1::DST1, sys2::DST2) where {DST1<:DescriptorStateSpace, DST2<:DescriptorStateSpace}
#     # fieldnames(DST1) == fieldnames(DST2) || (return false)
#     return all(getfield(sys1, f) == getfield(sys2, f) for f in fieldnames(DST1))
# end

# function isapprox(sys1::DST1, sys2::DST2; atol = zero(real(eltype(sys1))), 
#                   rtol = rtol::Real =  ((max(size(sys1.A)...))+1)*eps(real(float(one(real(eltype(sys1))))))*iszero(atol)) where 
#                   {DST1<:DescriptorStateSpace,DST2<:DescriptorStateSpace}
#     #fieldnames(DST1) == fieldnames(DST2) || (return false)
#     return all(isapprox(getfield(sys1, f), getfield(sys2, f); atol = atol, rtol = rtol) for f in fieldnames(DST1))
# end

# Base.copy(sys::DescriptorStateSpace{T}) where T = DescriptorStateSpace{T}(copy(sys.A), copy(sys.E), copy(sys.B), copy(sys.C), copy(sys.D), sys.Ts)


# # sum sys1+sys2
# function +(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
#     sys1.nu == 1 && sys1.ny == 1 && (sys2.ny > 1 || sys2.nu > 1) && (return ones(T1,sys2.ny,1)*sys1*ones(T1,1,sys2.nu) + sys2)
#     sys2.nu == 1 && sys2.ny == 1 && (sys1.nu > 1 || sys1.ny > 1) && (return sys1 + ones(T1,sys1.ny,1)*sys2*ones(T1,1,sys1.nu))
#     #Ensure systems have same dimensions and sampling times
#     size(sys1) == size(sys2) || error("The systems have different shapes.")
#     Ts = promote_Ts(sys1.Ts,sys2.Ts)

#     T = promote_type(T1, T2)
#     n1 = sys1.nx
#     n2 = sys2.nx
#     A = [sys1.A  zeros(T,n1,n2);
#          zeros(T,n2,n1) sys2.A]
#     B = [sys1.B ; sys2.B]
#     C = [sys1.C sys2.C;]
#     D = [sys1.D + sys2.D;]
#     if sys1.E == I && sys2.E == I
#         E = I
#     else
#         E = [sys1.E  zeros(T,n1,n2);
#              zeros(T,n2,n1) sys2.E]
#     end

#     return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
# end
# # difference sys1-sys2
# function -(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
#     sys1.nu == 1 && sys1.ny == 1 && (sys2.ny > 1 || sys2.nu > 1) && (return ones(T1,sys2.ny,1)*sys1*ones(T1,1,sys2.nu) - sys2)
#     sys2.nu == 1 && sys2.ny == 1 && (sys1.nu > 1 || sys1.ny > 1) && (return sys1 - ones(T1,sys1.ny,1)*sys2*ones(T1,1,sys1.nu))
#     #Ensure systems have same dimensions and sampling times
#     size(sys1) == size(sys2) || error("The systems have different shapes.")
#     Ts = promote_Ts(sys1.Ts,sys2.Ts)

#     T = promote_type(T1, T2)
#     n1 = sys1.nx
#     n2 = sys2.nx
 
#     A = [sys1.A  zeros(T,n1,n2);
#          zeros(T,n2,n1) sys2.A]
#     B = [sys1.B ; sys2.B]
#     C = [sys1.C -sys2.C;]
#     D = [sys1.D - sys2.D;]
#     if sys1.E == I && sys2.E == I
#         E = I
#     else
#         E = [sys1.E  zeros(T,n1,n2);
#              zeros(T,n2,n1) sys2.E]
#     end

#     return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
# end
# # negation -sys
# function -(sys::DescriptorStateSpace{T}) where T
#     return DescriptorStateSpace{T}(sys.A, sys.E, sys.B, -sys.C, -sys.D, sys.Ts)
# end
# # sys+mat and mat+sys
# function +(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
#     p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
#     size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T) + mat, sys.Ts)
# end
# +(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(sys,mat)
# # sys+I and I+sys
# function +(sys::DescriptorStateSpace{T1}, mat::UniformScaling{T2}) where {T1,T2}
#     size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T) + mat, sys.Ts)
# end

# # sys-mat and mat-sys
# function -(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
#     p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
#     size(sys) == (p, m) || error("The input-output dimensions of system does not match the shape of matrix.")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T) - mat, sys.Ts)
# end
# -(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(-sys,mat)

# # sys-I and I-sys
# function -(sys::DescriptorStateSpace{T1}, mat::UniformScaling{T2}) where {T1,T2}
#     size(sys,1) == size(sys,2) || error("The system must have the same number of inputs and outputs")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T) - mat, sys.Ts)
# end
# +(mat::UniformScaling{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(sys,mat)
# -(mat::UniformScaling{T1}, sys::DescriptorStateSpace{T2}) where {T1,T2} = +(-sys,mat)
# # sys ± n and n ± sys
# function +(sys::DescriptorStateSpace{T1}, n::Number) where T1 
#     T = promote_type(T1, eltype(n))
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T) .+ n, sys.Ts)
# end

# +(n::Number, sys::DescriptorStateSpace{T1}) where T1 = +(sys, n)
# -(n::Number, sys::DescriptorStateSpace{T1}) where T1 = +(-sys, n)
# -(sys::DescriptorStateSpace{T1},n::Number) where T1 = +(sys, -n)

# # multiplication sys1*sys2
# function *(sys1::DescriptorStateSpace{T1}, sys2::DescriptorStateSpace{T2}) where {T1,T2}
#     sys1.nu == 1 && sys1.ny == 1 && sys2.ny > 1 && (return dsdiag(sys1,sys2.ny)*sys2)
#     sys2.nu == 1 && sys2.ny == 1 && sys1.nu > 1 && (return sys1*dsdiag(sys2,sys1.nu))
#     sys1.nu == sys2.ny || error("sys1 must have same number of inputs as sys2 has outputs")
#     Ts = promote_Ts(sys1.Ts, sys2.Ts)
#     T = promote_type(T1, T2)
#     n1 = sys1.nx
#     n2 = sys2.nx

#     A = [sys1.A    sys1.B*sys2.C;
#          zeros(T,n2,n1)   sys2.A]
#     B = [sys1.B*sys2.D ; sys2.B]
#     C = [sys1.C   sys1.D*sys2.C;]
#     D = [sys1.D*sys2.D;]
#     if sys1.E == I && sys2.E == I
#         E = I
#     else
#         E = [sys1.E  zeros(T,n1,n2);
#              zeros(T,n2,n1) sys2.E]
#     end

#     return DescriptorStateSpace{T}(A, E, B, C, D, Ts)
# end
# # sys*mat
# function *(sys::DescriptorStateSpace{T1}, mat::VecOrMat{T2}) where {T1,T2}
#     p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
#     sys.nu == p || error("The input dimension of system does not match the number of rows of the matrix.")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), 
#                                    sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T)*to_matrix(T,mat), copy_oftype(sys.C,T), 
#                                    copy_oftype(sys.D,T)*to_matrix(T,mat), sys.Ts)
# end
# # mat*sys
# function *(mat::VecOrMat{T1}, sys::DescriptorStateSpace{T2}) where {T1 <: Number,T2}
#     p, m = typeof(mat) <: Vector ? (length(mat),1) : size(mat)
#     sys.ny == m || error("The output dimension of system does not match the number of columns of the matrix.")
#     T = promote_type(T1, T2)
#     return DescriptorStateSpace{T}(copy_oftype(sys.A,T), 
#                                    sys.E == I ? I : copy_oftype(sys.E,T),
#                                    copy_oftype(sys.B,T), to_matrix(T,mat)*copy_oftype(sys.C,T), 
#                                    to_matrix(T,mat)*copy_oftype(sys.D,T), sys.Ts)
# end
# # sI*sys
# function *(s::Union{UniformScaling,Number}, sys::DescriptorStateSpace{T}) where T
#     T1 = promote_type(eltype(s),T)
#     return DescriptorStateSpace{T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), copy_oftype(sys.B,T1), 
#                                    lmul!(s,copy_oftype(sys.C,T1)), lmul!(s,copy_oftype(sys.D,T1)), sys.Ts)

# end
# # sys*sI
# function *(sys::DescriptorStateSpace{T},s::Union{UniformScaling,Number}) where T
#     T1 = promote_type(eltype(s),T)
#     return DescriptorStateSpace{T1}(copy_oftype(sys.A,T1), sys.E == I ? I : copy_oftype(sys.E,T1), rmul!(copy_oftype(sys.B,T1),s), 
#                                    copy_oftype(sys.C,T1), rmul!(copy_oftype(sys.D,T1),s), sys.Ts)

# end

# # right division sys1/sys2 = sys1*inv(sys2)
# /(n::Union{UniformScaling,Number}, sys::DescriptorStateSpace) = n*inv(sys)
# /(sys::DescriptorStateSpace, n::Number) = sys*(1/n)
# /(sys::DescriptorStateSpace, n::UniformScaling) = sys*(1/n.λ)

# # left division sys1\sys2 = inv(sys1)*sys2
# \(n::Number, sys::DescriptorStateSpace) = (1/n)*sys
# \(n::UniformScaling, sys::DescriptorStateSpace) = (1/n.λ)*sys
# \(sys::DescriptorStateSpace, n::Union{UniformScaling,Number}) = inv(sys)*n

# # promotions
# Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DescriptorStateSpace, P <: Number } = DST
# Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DescriptorStateSpace, P <: UniformScaling } = DST
# Base.promote_rule(::Type{DST}, ::Type{P}) where {DST <: DescriptorStateSpace, P <: Matrix{<:Number} } = DST
# #Base.promote_rule(::Type{AbstractVecOrMat{<:Number}}, ::Type{DST}) where {DST <: DescriptorStateSpace} = DST
# #Base.promote_rule(::Type{DST}, ::Type{Union{AbstractVecOrMat{<:Number}, Number, UniformScaling}}) where {DST <: DescriptorStateSpace} = DST
# #Base.promote_rule(::Type{Union{AbstractVecOrMat{<:Number}, Number, UniformScaling}}, ::Type{DST}) where {DST <: DescriptorStateSpace} = DST
# # Base.promote_rule(::Type{DST}, ::Type{UniformScaling}) where {DST <: DescriptorStateSpace, P <: Number} = DST
# # Base.promote_rule(::Type{P}, ::Type{DST}) where {DST <: DescriptorStateSpace, P <: Number} = DST


# # conversions
# function Base.convert(::Type{DST}, p::Number) where {T, DST <: DescriptorStateSpace{T}}
#     T1 = promote_type(eltype(p),T)
#     dss(T1(p), Ts = sampling_time(DST))
# end
# sampling_time(::Type{DST}) where {DST <: DescriptorStateSpace} = 0.

 
