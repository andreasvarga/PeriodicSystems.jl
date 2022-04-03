# discrete-time case
"""
    PeriodicMatrix(M, T) -> A::PeriodicMatrix

Discrete-time periodic matrix representation. 

The discrete-time periodic matrix object `A` is built from a 
`p`-vector `M` of real matrices, the associated time period `T` and 
the number of subperiods specified via the keyword argument `nperiod = k`. 

`M` contains the cyclic component matrices `M[i]`, `i = 1,..., p`, 
where `M[i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T/k`, with `Δ := T/(kp)`, the associated sample time. 
It is assumed that `M[k] := M[mod(k-1,p)+1]` for arbitrary `k`. 
All component matrices are allowed to have arbitrary (time-varying) dimensions.
The component matrices `M`, the period `T`, the discrete period `p` and the sample time `Δ`
can be accessed via `A.M`, `A.period`, `A.dperiod` and `A.Ts`, respectively. 
"""
struct  PeriodicMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T}
    M::Vector{Matrix{T}}
    period::Float64
    nperiod::Int
end 
# additional constructors
function  PeriodicMatrix(M::Vector{MT}, period::Real; nperiod::Int = 1) where {MT <: Array} 
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   any(ndims.(M) .> 2) && error("only vectors with vector or matrix elements supported")
   m = size.(M,2)
   T = promote_type(eltype.(M)...)
   return any(m .== 1) ?  PeriodicMatrix{:d,T}([T.(reshape(M[i],size(M[i],1),m[i])) for i in 1:length(M)], Float64(period), nperiod)  :  
                   PeriodicMatrix{:d,T}([T.(M[i]) for i in 1:length(M)], Float64(period), nperiod) 
end
PeriodicMatrix{:d,T}(A::Vector{Matrix{T}}, period::Real; nperiod::Int = 1) where {T} = 
      PeriodicMatrix{:d,T}(A, period, nperiod)

# period change
function PeriodicMatrix{:d,T}(A::PeriodicMatrix{:d,T1}, period::Real) where {T,T1}
   period > 0 || error("period must be positive") 
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicMatrix{:d,T}([T.(A.M[i]) for i in 1:length(A)], Aperiod*d, A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicMatrix{:d,T}([T.(A.M[i]) for i in 1:length(A)], Aperiod/n, nperiod)
   end
end

# function PeriodicMatrix{:d,T}(A::Vector{Matrix{T1}}, period::Real) where {T,T1}
#     PeriodicMatrix([T.(A[i]) for i in 1:length(A)], period)
# end
# PeriodicMatrix(M::Vector{MT}, period::Real) where {T <: Real, MT <: Array{T}} = 
#            PeriodicMatrix{:d,T}(M, period)
PeriodicMatrix(M::Vector{Matrix{T}}, period::Real; nperiod::Int = 1) where {T <: Real} = 
       PeriodicMatrix{:d,T}(M, period, nperiod)
PeriodicMatrix(M::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real} =
   PeriodicMatrix{:d,T}([reshape(M,size(M,1),size(M,2))], period, nperiod)
function Base.getproperty(A::PeriodicMatrix, d::Symbol)  
   if d === :dperiod
      return length(getfield(A, :M))
   elseif d === :Ts
      return A.period/A.dperiod/A.nperiod
   else
      getfield(A, d)
   end
end
Base.propertynames(A::PeriodicMatrix) = (:dperiod, :Ts, fieldnames(typeof(A))...)
isconstant(A::PeriodicMatrix) = (A.dperiod == 1)
Base.size(A::PeriodicMatrix) = (size.(A.M,1),size.(A.M,2))
Base.size(A::PeriodicMatrix, d::Integer) = size.(A.M,d)
Base.length(A::PeriodicMatrix) = A.dperiod
Base.eltype(A::PeriodicMatrix{:d,T}) where T = T
"""
    PeriodicArray(M, T; nperiod = k) -> A::PeriodicArray

Discrete-time periodic array representation.

The discrete-time periodic array object `A` is built from a `m×n×p` real array
`M`, the associated time period `T` and the number of subperiods specified via 
the keyword argument `nperiod = k`. 
`M` contains the cyclic component matrices `M[:,:,i]`, `i = 1,..., p`, 
where `M[:,:,i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T/k`, with `Δ := T/(kp)`, the associated sample time. 
It is assumed that  `M[:,:,k] := M[:,:,mod(k-1,p)+1]` for arbitrary `k`. 
The component matrices `M`, the period `T`, the discrete period `p` and the sample time `Δ`
can be accessed via `A.M`, `A.period`, `A.dperiod` and `A.Ts`, respectively. 
"""
struct PeriodicArray{Domain,T} <: AbstractPeriodicArray{Domain,T}
    M::Array{T,3}
    period::Float64
    nperiod::Int
end 
# additional constructors
function  PeriodicArray{:d,T}(M::Array{T,3}, period::Real; nperiod::Int = 1) where {T <: Real} 
   period > 0 || error("period must be positive")       
   nperiod > 0 || error("number of subperiods must be positive") 
   PeriodicArray{:d,T}(M, Float64(period), nperiod) 
end
function PeriodicArray{:d,T}(A::PeriodicArray{:d,T1}, period::Real) where {T,T1}
   period > 0 || error("period must be positive") 
   #isconstant(A) && (return PeriodicArray{:d,T}(convert(Array{T,3},A.M), period; nperiod = 1))
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicArray{:d,T}(convert(Array{T,3},A.M), Aperiod*d; nperiod = A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicArray{:d,T}(convert(Array{T,3},A.M), Aperiod/n; nperiod)
   end
end
#PeriodicArray{:d,T}(M::Array{T1,3}, period::Real; nperiod::Int = 1) where {T,T1} = PeriodicArray(T.(M), period; nperiod)
PeriodicArray(M::Array{T,3}, period::Real; nperiod::Int = 1) where {T <: Real} = PeriodicArray{:d,T}(M, period, nperiod)
PeriodicArray(M::VecOrMat{T}, period::Real; nperiod::Int = 1) where T = PeriodicArray(reshape(M,size(M,1),size(M,2),1), period; nperiod)
function Base.getproperty(A::PeriodicArray, d::Symbol)  
   if d === :dperiod
      return size(getfield(A, :M), 3)
   elseif d === :Ts
      return A.period/A.dperiod/A.nperiod
   else
      getfield(A, d)
   end
end
Base.propertynames(A::PeriodicArray) = (:dperiod, :Ts, fieldnames(typeof(A))...)
isconstant(A::PeriodicArray) = (A.dperiod == 1)
Base.size(A::PeriodicArray) = (size(A.M,1),size(A.M,2))
Base.size(A::PeriodicArray, d::Integer) = size(A.M,d)
Base.length(A::PeriodicArray) = A.dperiod
Base.eltype(A::PeriodicArray{:d,T}) where T = T

"""
    PeriodicFunctionMatrix(f, T) -> A::PeriodicFunctionMatrix

Continuous-time periodic function matrix representation.

The continuous-time periodic function matrix object `A` is built from a 
time periodic real matrix function `f(t)` of real time variable `t` 
and the associated time period `T`. 
It is assumed that  `F(t) = F(t+T)` for any real time value `t`.
The function `f(t)`, the period `T`, and the row and column dimensions 
of `f(t)` can be accessed via `A.f`, `A.period` and the tuple `A.dims`, respectively.
"""
struct PeriodicFunctionMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T}
   f::Function
   period::Float64
   dims::Tuple{Int,Int}
   _isconstant::Bool
end 
# additional constructors
function PeriodicFunctionMatrix{:c,Tf}(f::Function, period::Real; isconst::Bool = false) where {Tf} 
   period > 0 || error("period must be positive") 
   F0 = f(period)
   nd = ndims(F0)
   nd == 2 || error("two-dimensional function array expected, got an $nd -dimensional array")
   eltype(F0) == Tf ? PeriodicFunctionMatrix{:c,Tf}(t -> f(t), Float64(period), size(F0), isconst) :
                      PeriodicFunctionMatrix{:c,Tf}(t -> convert(Matrix{Tf},f(Tf(t))), Float64(period), size(F0), isconst)
end
PeriodicFunctionMatrix(f::F, period::Real; isconst::Bool = false) where {F<:Function}  = 
             PeriodicFunctionMatrix{:c,eltype(f(period))}(f, period; isconst)
PeriodicFunctionMatrix(A::VecOrMat{T}, period::Real) where {T <: Real}  = 
          PeriodicFunctionMatrix{:c,T}(t -> reshape(A,size(A,1),size(A,2)), period; isconst = true) 
function PeriodicFunctionMatrix(at::PeriodicFunctionMatrix, period::Real = at.period; isconst::Bool = isconstant(at))
   # at.period = period
   # at._isconstant = isconst
   # return at
   return PeriodicFunctionMatrix(at.f, period; isconst)
end
# properties
isconstant(A::PeriodicFunctionMatrix) = A._isconstant
function isperiodic(f::Function, period::Real)  
   t = rand(typeof(period))
   return f(t) ≈ f(t+period)
end
isperiodic(A::PeriodicFunctionMatrix) = isconstant(A) ? true : isperiodic(A.f,A.period)
Base.size(A::PeriodicFunctionMatrix) = A.dims
Base.size(A::PeriodicFunctionMatrix, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(A::PeriodicFunctionMatrix{:c,T}) where T = T

"""
    PeriodicSymbolicMatrix(F, T) -> A::PeriodicSymbolicMatrix

Continuous-time periodic symbolic matrix representation.
 
The continuous-time periodic symbolic matrix object `A` is built from `F`, a 
symbolic periodic real matrix or vector of symbolic variable `t`, and the associated time period `T`. 
It is assumed that  `F(t) = F(t+T)` for any real time value `t`.
The symbolic matrix `F` and the period `T` can be accessed via `A.F` and `A.period`, respectively.
"""
struct PeriodicSymbolicMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   F::Matrix{<:Num}
   period::Float64
   function  PeriodicSymbolicMatrix{:c,T}(F::VecOrMat{T}, period::Real) where {T <: Num} 
      period > 0 || error("period must be positive")       
      # check that array F is depending only on t
      tt = rand()
      @variables t
      Ft = substitute.(F, (Dict(t => tt),))
      m, n = size(Ft,1), size(Ft,2)
      any(length.(Symbolics.get_variables.(Ft)) .> 0 ) && error("t must be the only variable in F")
      new{:c,T}(n == 1 ? reshape(F,m,n) : F, Float64(period)) 
   end
end 
# additional constructors
PeriodicSymbolicMatrix(F::VecOrMat{T}, period::Real) where {T <: Num}  = 
             PeriodicSymbolicMatrix{:c,T}(F, period)
PeriodicSymbolicMatrix(A::PeriodicSymbolicMatrix{:c,T}, period::Real = A.period) where {T <: Num}  = 
             PeriodicSymbolicMatrix{:c,T}(A.F, period)
# properties 
isconstant(A::PeriodicSymbolicMatrix) = all(length.(Symbolics.get_variables.(A.F)) .== 0)
function isperiodic(A::VecOrMat{T}, period::Real) where {T <: Num} 
   tt = rand()
   @variables t
   At = substitute.(A, (Dict(t => tt),))
   return norm(At - substitute.(A, (Dict(t => tt+period),))) <= eps(10.)*max(1.,norm(At)) 
end
isperiodic(A::PeriodicSymbolicMatrix) = isconstant(A) ? true : isperiodic(A.F,A.period)
Base.size(A::PeriodicSymbolicMatrix) = size(A.F)
Base.size(A::AbstractPeriodicArray, d::Integer) = d <= 2 ? size(A)[d] : 1

struct HarmonicArray{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   values::Array{Complex{T},3}
   period::Float64
   nperiod::Int
end
# additional constructors
"""
     HarmonicArray(Ahr, T; nperiod = k) -> A::HarmonicArray

Continuous-time harmonic array representation.

The harmonic array object `A` of period `T` is built using
the harmonic representation of a periodic matrix `Ahr(t)` of subperiod `T′ = T/k` in the form

                     p
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T′)+As_i*sin(i*2*π*t/T′) ) ,
                    i=1 

where `k ≥ 1` is the number of subperiods (default: `k = 1`).                   
The `m×n×(p+1)` complex array `Ahr` contains the harmonic components as follows:
`Ahr[:,:,1]` contains the constant term `A_0` (the mean value) and
the real and imaginary parts of `Ahr[:,:,i+1]`  
for `i = 1, ..., p` contain the coefficient matrices `Ac_i` and `As_i`, respectively. 
The complex matrix `Ahr` containing the harmonic components, the period `T` and the 
number of subperiods `k` can be accessed via `A.values`, `A.period` and `A.nperiod`, respectively.
"""
function HarmonicArray{:c,T}(Ahr::Array{Complex{T1},3}, period::Real; nperiod::Int = 1) where {T,T1}
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   (size(Ahr,3) > 0 && iszero(imag(view(Ahr,:,:,1)))) || error("imaginary part of constant term must be zero")
   HarmonicArray{:c,T}(convert(Array{Complex{T},3},Ahr), Float64(period), nperiod) 
end
HarmonicArray(Ahr::Array{Complex{T},3}, period::Real; nperiod::Int = 1) where T = HarmonicArray{:c,T}(Ahr, period; nperiod)
function HarmonicArray{:c,T}(A::HarmonicArray{:c,T1}, period::Real) where {T,T1}
   period > 0 || error("period must be positive") 
   isconstant(A) && (return HarmonicArray{:c,T}(convert(Array{Complex{T},3},A.values), period; nperiod = 1))
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      HarmonicArray{:c,T}(convert(Array{Complex{T},3},A.values), Aperiod*d; nperiod = A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      HarmonicArray{:c,T}(convert(Array{Complex{T},3},A.values), Aperiod/n; nperiod)
   end
end
"""
     HarmonicArray(A0, Ac, As, T) -> A::HarmonicArray

Construct a harmonic array representation from the harmonic components.

The harmonic array object `A` is built for 
the harmonic representation `Ahr(t)` of a periodic matrix of period `T` in the form

                     p
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T)+As_i*sin(i*2*π*t/T) ) ,
                    i=1 

where the constant term `A_0` is contained in the real matrix `A0`, and `Ac` and `As` are
vectors of real matrices such that the `i`-th (cosinus) coefficient matrix 
`Ac_i` is contained in `Ac[i]` and the `i`-th (sinus) coefficient matrix 
`As_i` is contained in `As[i]`. `p` is the maximum of length of the vectors of matrices `Ac` and `As`. 
If the length of `Ac` or `As` is less than `p`, then zero trailing matrices are assumed in the respective matrix. 
All component matrices must have the same dimensions.
The complex matrix containing the harmonic components and the period `T` 
can be accessed via `A.values` and `A.period`, respectively.
"""
function HarmonicArray(A0::MT, Acos::Union{Vector{MT},Nothing}, 
                               Asin::Union{Vector{MT},Nothing}, period::Real) where {T <: Real, MT <: VecOrMat{T}}
   nc = isnothing(Acos) ? 0 : length(Acos)
   ns = isnothing(Asin) ? 0 : length(Asin)
   nmin = min(nc,ns)
   N = max(1,max(nc,ns)+1)
   ahr = Array{Complex{T},3}(undef, size(A0,1), size(A0,2), N)
   ahr[:,:,1] = A0
   [ahr[:,:,i+1] = complex.(Acos[i],Asin[i]) for i in 1:nmin]
   [ahr[:,:,i+1] = Acos[i] for i in nmin+1:nc]
   [ahr[:,:,i+1] = im*Asin[i] for i in nmin+1:ns]
   #HarmonicArray(ahr, period)
   HarmonicArray{:c,T}(ahr, period)
end
HarmonicArray(A0::VecOrMat{T}, period::Real) where {T <: Real}  = 
          HarmonicArray(complex(reshape(A0,size(A0,1),size(A0,2),1)), period) 
HarmonicArray(A0::VecOrMat{T}, Acos::Vector{MT}, period::Real) where {T <: Real, MT <: VecOrMat{T}}  = 
          HarmonicArray(A0, Acos, nothing, period) 

# properties
isconstant(A::HarmonicArray) = size(A.values,3) <= 1
isperiodic(A::HarmonicArray) = true
Base.size(A::HarmonicArray) = (size(A.values,1),size(A.values,2))
Base.eltype(A::HarmonicArray{:c,T}) where T = T

"""
    PeriodicTimeSeriesMatrix(At, T) -> A::PeriodicTimeSeriesMatrix

Continuous-time periodic time series matrix representation.

The continuous-time periodic time series matrix object `A` is built from a 
`p`-vector `At` of real matrices and the associated time period `T`. 
`At` contains the cyclic component matrices `At[i]`, `i = 1,..., p`, 
where `At[i]` represents the value `A(Δ(i-1))` of a time periodic matrix `A(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that `At[k] := At[mod(k-1,p)+1]` for arbitrary `k`. 
All component matrices must have the same dimensions.
The component matrices `At` and the period `T` 
can be accessed via `A.values` and `A.period`, respectively. 
"""
struct PeriodicTimeSeriesMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   values::Vector{Array{T,2}}
   period::Float64
   nperiod::Int
end
# additional constructors
function PeriodicTimeSeriesMatrix{:c,T}(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real; nperiod::Int = 1) where {T <: Real} 
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   N = length(At) 
   #N <= 1 && (return PeriodicTimeSeriesMatrix{:c,T}(At, Float64(period); nperiod) ) # the constant matrix case 
   n1, n2 = size(At[1],1), size(At[1],2)
   (all(size.(At,1) .== n1) && all(size.(At,2) .== n2)) || error("all component matrices must have the same dimensions")

   # adjust final data to matrix type
   PeriodicTimeSeriesMatrix{:c,T}(n2 == 1 ? [reshape(At[j],n1,n2) for j in 1:N] : At, Float64(period), nperiod) 
end
PeriodicTimeSeriesMatrix{:c,T}(A::Vector{Matrix{T1}}, period::Real; nperiod::Int = 1) where {T,T1} = 
   PeriodicTimeSeriesMatrix([T.(A[i]) for i in 1:length(A)], period; nperiod)
PeriodicTimeSeriesMatrix(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real; nperiod::Int = 1) where {T <: Real} = 
     PeriodicTimeSeriesMatrix{:c,T}(At, period; nperiod)  
PeriodicTimeSeriesMatrix(At::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real}  = 
        PeriodicTimeSeriesMatrix([reshape(At,size(At,1),size(At,2))], period; nperiod) 
# period change
function PeriodicTimeSeriesMatrix{:c,T}(A::PeriodicTimeSeriesMatrix{:c,T1}, period::Real) where {T,T1}
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicTimeSeriesMatrix{:c,T}([T.(A.values[i]) for i in 1:length(A)], Aperiod*d; nperiod = A.nperiod*d)      
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicTimeSeriesMatrix{:c,T}([T.(A.values[i]) for i in 1:length(A)], Aperiod/n; nperiod)
   end
end

# properties
isconstant(At::PeriodicTimeSeriesMatrix) = length(At.values) <= 1
isperiodic(At::PeriodicTimeSeriesMatrix) = true
Base.length(At::PeriodicTimeSeriesMatrix) = length(At.values) 
Base.size(At::PeriodicTimeSeriesMatrix) = length(At) > 0 ? size(At.values[1]) : (0,0)
Base.eltype(At::PeriodicTimeSeriesMatrix{:c,T}) where T = T

# conversions to discrete-time PeriodicMatrix
Base.convert(::Type{PeriodicMatrix}, A::PeriodicArray{:d,T}) where T = 
             PeriodicMatrix{:d,T}([A.M[:,:,i] for i in 1:size(A.M,3)],A.period; nperiod = A.nperiod)

# conversions to discrete-time PeriodicArray
# function Base.convert(::Type{PeriodicArray}, A::PeriodicMatrix{:d,T}) where T
#     N = length(A)
#     m, n = size.(A,1), size.(A,2)
#     N == 0 && PeriodicArray(Array{T,3}(undef,0,0,0),A.period)
#     if any(m .!= m[1]) || any(m .!= m[1]) 
#        @warn "Non-constant dimensions: the resulting component matrices padded with zeros"
#        t = zeros(T,maximum(m),maximum(n),N)
#        [copyto!(view(t,1:m[i],1:n[i],i),A[i]) for i in 1:N]
#        PeriodicArray{:d,T}(t,A.period)
#     else
#        t = zeros(T,m[1],n[1],N)
#        [copyto!(view(t,:,:,i),A[i]) for i in 1:N]
#        PeriodicArray{:d,T}(t,A.period)
#     end
# end
Base.convert(::Type{PeriodicArray}, A::PeriodicMatrix) where T = pm2pa(A)


# conversions to continuous-time PeriodicFunctionMatrix
function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicFunctionMatrix) where T
   return eltype(A) == T ? A : PeriodicFunctionMatrix(x -> T.(A.f(T(x))), A.period)
end

function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicSymbolicMatrix) where T
   @variables t
   f = eval(build_function(A.F, t, expression=Val{false})[1])
   PeriodicFunctionMatrix(x -> f(T(x)), A.period)
end
function Base.convert(::Type{PeriodicFunctionMatrix}, A::PeriodicSymbolicMatrix) where T
   @variables t
   f = eval(build_function(A.F, t, expression=Val{false})[1])
   PeriodicFunctionMatrix(x -> f(x), A.period)
end
# function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicSymbolicMatrix) where T
#    @variables t
#    f = eval(build_function(A.F, t, expression=Val{false})[1])
#    PeriodicFunctionMatrix(x -> f(x), A.period)
# end

# PeriodicFunctionMatrix(ahr::HarmonicArray, period::Real = ahr.period; exact = true)  = 
#           PeriodicFunctionMatrix(t::Real -> hreval(ahr,t;exact)[1], period) 
Base.convert(::Type{PeriodicFunctionMatrix}, ahr::HarmonicArray)  where T = 
         PeriodicFunctionMatrix(t::Real -> hreval(ahr,t), ahr.period)
Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, ahr::HarmonicArray)  where T = 
         PeriodicFunctionMatrix(t::Real -> hreval(ahr,T(t)), ahr.period)
Base.convert(::Type{PeriodicFunctionMatrix}, At::PeriodicTimeSeriesMatrix)  where T = 
    ts2pfm(At; method = "cubic")
# function PeriodicFunctionMatrix(A::PeriodicTimeSeriesMatrix; method = "linear")
#    N = length(A.values)
#    N == 0 && error("empty time array")
#    N == 1 && (return t -> A.values[1])
#    #dt = A.time[2]-A.time[1]
#    dt = A.period/N
#    ts = (0:N)*dt
#    if method == "linear"
#       itp = LinearInterpolation(ts, push!(copy(A.values),A.values[1]))
#       return PeriodicFunctionMatrix(t -> itp(mod(t, A.period)), A.period )
#    elseif method == "cubic"      
#       n1, n2 = size(A.values[1])
#       intparray = Array{Any,2}(undef,n1, n2)
#       [intparray[i,j] = CubicSplineInterpolation(ts,[getindex.(A.values,i,j);A.values[1][i,j]],bc=Line(OnGrid())) for i in 1:n1, j in 1:n2]
#       return PeriodicFunctionMatrix(t -> [intparray[i,j](mod(t, A.period)) for i in 1:n1, j in 1:n2 ], A.period )
#    end
# end

# conversions to continuous-time PeriodicSymbolicMatrix
function Base.convert(::Type{PeriodicSymbolicMatrix}, A::PeriodicFunctionMatrix) where T
   @variables t
   PeriodicSymbolicMatrix(Num.(A.f(t)), A.period)
end
function Base.convert(::Type{PeriodicSymbolicMatrix{:c,T}}, A::PeriodicFunctionMatrix) where T
   @variables t
   PeriodicSymbolicMatrix(Num.(A.f(t)), A.period)
end
Base.convert(::Type{PeriodicSymbolicMatrix}, ahr::HarmonicArray)  where T = 
   PeriodicSymbolicMatrix(hr2psm(ahr), ahr.period)
# function PeriodicSymbolicMatrix(ahr::HarmonicArray, period::Real = ahr.period)
#    @variables t
#    PeriodicSymbolicMatrix(hreval(ahr,t)[1], period)
# end


# conversion to continuous-time HarmonicArray
Base.convert(::Type{HarmonicArray}, A::PeriodicTimeSeriesMatrix) = ts2hr(A)
Base.convert(::Type{HarmonicArray}, A::PeriodicFunctionMatrix) = pfm2hr(A)
Base.convert(::Type{HarmonicArray}, A::PeriodicSymbolicMatrix) = psm2hr(A)

# conversions to PeriodicTimeSeriesMatrix
Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicFunctionMatrix) = 
         PeriodicTimeSeriesMatrix(A.f.((0:127)*A.period/128), A.period)
# conversions of discrete-time periodic matrices to continuous-time PeriodicTimeSeriesMatrix

# function PeriodicTimeSeriesMatrix(A::PeriodicMatrix{:d,T}, period::Real = A.period) where {T <: Real}
#    N = length(A.M) 
#    N <= 1 && (return PeriodicTimeSeriesMatrix(A.M, period))
#    m, n = size(A.M[1])
#    (any(size.(A.M,1) .!= m) || any(size.(A.M,2) .!= n)) && 
#          error("only periodic matrices with constant dimensions supported")
#    PeriodicTimeSeriesMatrix(A.M, period)
# end
Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicMatrix) =
    convert(PeriodicTimeSeriesMatrix, pm2pa(A))
Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicArray) =
    PeriodicTimeSeriesMatrix([A.M[:,:,i] for i in 1:size(A.M,3)], A.period; nperiod = A.nperiod)

