# discrete-time case
"""
    PeriodicMatrix(M, T) -> A::PeriodicMatrix

Discrete-time periodic matrix representation. 

The discrete-time periodic matrix object `A` is built from a 
`p`-vector `M` of real matrices and the associated time period `T`. 
`M` contains the cyclic component matrices `M[i]`, `i = 1,..., p`, 
where `M[i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that `M[k] := M[mod(k-1,p)+1]` for arbitrary `k`. 
All component matrices are allowed to have arbitrary (time-varying) dimensions.
The component matrices `M`, the period `T` and the discrete period `p`
can be accessed via `A.M`, `A.period` and `A.dperiod`, respectively. 
"""
struct  PeriodicMatrix{Domain,T} <: AbstractPeriodicArray{Domain}
    M::Vector{Matrix{T}}
    period::Float64
    function  PeriodicMatrix{:d,T}(M::Vector{MT}, period::Real) where {T <: Real, MT <: Array{T}} 
       period > 0 || error("period must be positive") 
       any(ndims.(M) .> 2) && error("only vectors with vector or matrix elements supported")
       m = size.(M,2)
       return any(m .== 1) ?  new{:d,T}(reshape.(M,size.(M,1),m), Float64(period))  :  
                       new{:d,T}(M, Float64(period)) 
    end
end 
PeriodicMatrix(M::Vector{MT}, period::Real) where {T <: Real, MT <: Array{T}} = 
           PeriodicMatrix{:d,T}(M, period)
PeriodicMatrix(M::VecOrMat{T}, period::Real) where {T <: Real} = PeriodicMatrix([M], period)
function Base.getproperty(A::PeriodicMatrix, d::Symbol)  
   if d === :dperiod
      return length(getfield(A, :M))
   else
      getfield(A, d)
   end
end
Base.propertynames(A::PeriodicMatrix) = (:dperiod, fieldnames(typeof(A))...)

"""
    PeriodicArray(M, T) -> A::PeriodicArray

Discrete-time periodic array representation.

The discrete-time periodic array object `A` is built from a `m×n×p` real array
`M` and the associated time period `T`. 
`M` contains the cyclic component matrices `M[:,:,i]`, `i = 1,..., p`, 
where `M[:,:,i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that  `M[:,:,k] := M[:,:,mod(k-1,p)+1]` for arbitrary `k`. 
The component matrices `M`, the period `T` and the discrete period `p`
can be accessed via `A.M`, `A.period` and `A.dperiod`, respectively. 
"""
struct PeriodicArray{Domain,T} <: AbstractPeriodicArray{Domain}
    M::Array{T,3}
    period::Float64
    function  PeriodicArray{:d,T}(M::Array{T,3}, period::Real) where {T <: Real} 
       period > 0 || error("period must be positive")       
       new{:d,T}(M, Float64(period)) 
    end
end 
PeriodicArray(M::Array{T,3}, period::Real) where {T <: Real} = PeriodicArray{:d,T}(M, period)
PeriodicArray(M::VecOrMat{T}, period::Real) where T = PeriodicArray(reshape(M,size(M,1),size(M,2),1), period)
function Base.getproperty(A::PeriodicArray, d::Symbol)  
   if d === :dperiod
      return size(getfield(A, :M), 3)
   else
      getfield(A, d)
   end
end
Base.propertynames(A::PeriodicArray) = (:dperiod, fieldnames(typeof(A))...)

"""
    PeriodicFunctionMatrix(f, T) -> A::PeriodicFunctionMatrix

Continuous-time periodic function matrix representation.

The continuous-time periodic function matrix object `A` is built from a 
time periodic real matrix function `f(t)` of real time variable `t` 
and the associated time period `T`. 
The periodicity condition `f(t) ≈ f(t+T)` is checked for a randomly generated value of `t`
and a warning is issued if this condition is not satisfied.
The function `f(t)`, the period `T`, and the row and column dimensions 
of `f(t)` can be accessed via `A.f`, `A.period` and the tuple `A.dims`, respectively.
"""
struct PeriodicFunctionMatrix{Domain,T} <: AbstractPeriodicArray{Domain}
   f::Function
   period::Float64
   dims::Tuple{Int,Int}
   function PeriodicFunctionMatrix{:c,Tf}(f::Function, period::Real) where {Tf} 
      period > 0 || error("period must be positive") 
      T = eltype(f(0))
      t = rand(T)
      Ft = f(t)
      nd = ndims(Ft)
      nd == 2 || error("two-dimensional function array expected, got an $nd -dimensional array")
      Ft ≈ f(t+period) || @warn "function f is likely not periodic"
      new{:c,T}(t -> f(t), Float64(period), size(Ft)) 
   end
end 
PeriodicFunctionMatrix(f::F, period::Real) where {F<:Function}  = 
             PeriodicFunctionMatrix{:c,eltype(f(0))}(f, period)
PeriodicFunctionMatrix(A::VecOrMat{T}, period::Real) where {T <: Real}  = 
          PeriodicFunctionMatrix(t -> reshape(A,size(A,1),size(A,2)), period) 
PeriodicFunctionMatrix(at::PeriodicFunctionMatrix, period::Real = at.period)  = 
          PeriodicFunctionMatrix(at.f, period)

"""
    PeriodicSymbolicMatrix(F, T) -> A::PeriodicSymbolicMatrix

Continuous-time periodic symbolic matrix representation.
 
The continuous-time periodic symbolic matrix object `A` is built from `F`, a 
symbolic periodic real matrix or vector of symbolic variable `t`, and the associated time period `T`. 
It is assumed that  `F(t) = F(t+T)` for any real time value `t`.
The symbolic matrix `F` and the period `T` can be accessed via `A.F` and `A.period`, respectively.
"""
struct PeriodicSymbolicMatrix{Domain,T} <: AbstractPeriodicArray{Domain} 
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
      # check periodicity
      norm(Ft - substitute.(F, (Dict(t => tt+period),))) <= eps(10.)*max(1.,norm(Ft))  || 
         @warn "Matrix likely not periodic in variable t with period $period"
      new{:c,T}(n == 1 ? reshape(F,m,n) : F, Float64(period)) 
   end
end 
PeriodicSymbolicMatrix(F::VecOrMat{T}, period::Real) where {T <: Num}  = 
             PeriodicSymbolicMatrix{:c,T}(F, period)
PeriodicSymbolicMatrix(A::PeriodicSymbolicMatrix{:c,T}, period::Real = A.period) where {T <: Num}  = 
             PeriodicSymbolicMatrix{:c,T}(A.F, period)

"""
     HarmonicArray(Ahr, T) -> A::HarmonicArray

Continuous-time harmonic array representation.

The harmonic array object `A` is built for 
the harmonic representation of a periodic matrix `Ahr(t)` of period `T` in the form

                     p
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T)+As_i*sin(i*2*π*t/T) ) .
                    i=1 

The `m×n×(p+1)` complex array `Ahr` contains the harmonic components as follows:
`Ahr[:,:,1]` contains the constant term `A_0` (the mean value) and
the real and imaginary parts of `Ahr[:,:,i+1]`  
for `i = 1, ..., p` contain the coefficient matrices `Ac_i` and `As_i`, respectively. 
The complex matrix `Ahr` containing the harmonic components and the period `T` 
can be accessed via `A.values` and `A.period`, respectively.
"""
struct HarmonicArray{Domain,T} <: AbstractPeriodicArray{Domain} 
   values::Array{Complex{T},3}
   period::Float64
   function HarmonicArray{:c,T}(Ahr::Array{Complex{T},3}, period::Real) where T
      period > 0 || error("period must be positive") 
      (size(Ahr,3) > 0 && iszero(imag(view(Ahr,:,:,1)))) || error("imaginary part of constant term must be zero")
      new{:c,T}(Ahr, Float64(period)) 
   end
end
HarmonicArray(Ahr::Array{Complex{T},3}, period::Real) where {T} = HarmonicArray{:c,T}(Ahr, period) 
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
function HarmonicArray(A0::MT, Acos::Union{Vector{MT},Vector{Any}}, 
                               Asin::Union{Vector{MT},Vector{Any}}, period::Real) where {T <: Real, MT <: VecOrMat{T}}
   nc = isnothing(Acos) ? 0 : length(Acos)
   ns = isnothing(Asin) ? 0 : length(Asin)
   nmin = min(nc,ns)
   N = max(1,max(nc,ns)+1)
   ahr = Array{ComplexF64,3}(undef, size(A0,1), size(A0,2), N)
   ahr[:,:,1] = A0
   [ahr[:,:,i+1] = complex.(Acos[i],Asin[i]) for i in 1:nmin]
   [ahr[:,:,i+1] = Acos[i] for i in nmin+1:nc]
   [ahr[:,:,i+1] = im*Asin[i] for i in nmin+1:ns]
   HarmonicArray(ahr, period)
end
HarmonicArray(A0::VecOrMat{T}, period::Real) where {T <: Real}  = 
          HarmonicArray(complex(reshape(A0,size(A0,1),size(A0,2),1)), period) 
HarmonicArray(A0::VecOrMat{T}, Acos::Vector{MT}, period::Real) where {T <: Real, MT <: VecOrMat{T}}  = 
          HarmonicArray(A0, Acos, nothing, period) 

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
struct PeriodicTimeSeriesMatrix{Domain,T} <: AbstractPeriodicArray{Domain} 
   values::Vector{Array{T,2}}
   period::Float64
   function PeriodicTimeSeriesMatrix{:c,T}(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real) where {T <: Real} 
      period > 0 || error("period must be positive") 
      N = length(At) 
      N <= 1 && (return new{:c,T}(At, Float64(period)) ) # the constant matrix case 
      n1, n2 = size(At[1],1), size(At[1],2)
      (all(size.(At,1) .== n1) && all(size.(At,2) .== n2)) || error("all component matrices must have the same dimensions")

      # adjust final data to matrix type
      new{:c,T}(n2 == 1 ? [reshape(At[j],n1,n2) for j in 1:N] : At, Float64(period)) 
   end
end
PeriodicTimeSeriesMatrix(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real) where {T <: Real} = 
    PeriodicTimeSeriesMatrix{:c,T}(At, period)  
PeriodicTimeSeriesMatrix(At::VecOrMat{T}, period::Real) where {T <: Real}  = 
        PeriodicTimeSeriesMatrix([reshape(At,size(At,1),size(At,2))], period) 

# conversions to discrete-time PeriodicMatrix
Base.convert(::Type{PeriodicMatrix}, A::PeriodicArray{:d,T}) where T = 
             PeriodicMatrix{:d,T}([A.M[:,:,i] for i in 1:size(A.M,3)],A.period)

# conversions to discrete-time PeriodicArray
function Base.convert(::Type{PeriodicArray}, A::PeriodicMatrix{:d,T}) where T
    N = length(A)
    m, n = size.(A,1), size.(A,2)
    N == 0 && PeriodicArray(Array{T,3}(undef,0,0,0),A.period)
    if any(m .!= m[1]) || any(m .!= m[1]) 
       @warn "Non-constant dimensions: the resulting component matrices padded with zeros"
       t = zeros(T,maximum(m),maximum(n),N)
       [copyto!(view(t,1:m[i],1:n[i],i),A[i]) for i in 1:N]
       PeriodicArray{:d,T}(t,A.period)
    else
       t = zeros(T,m[1],n[1],N)
       [copyto!(view(t,:,:,i),A[i]) for i in 1:N]
       PeriodicArray{:d,T}(t,A.period)
    end
end

# conversions to continuous-time PeriodicFunctionMatrix
function Base.convert(::Type{PeriodicFunctionMatrix}, A::PeriodicSymbolicMatrix) 
   @variables t
   f = eval(build_function(A.F, t, expression=Val{false})[1])
   PeriodicFunctionMatrix(x -> f(x), A.period)
end
# PeriodicFunctionMatrix(ahr::HarmonicArray, period::Real = ahr.period; exact = true)  = 
#           PeriodicFunctionMatrix(t::Real -> hreval(ahr,t;exact)[1], period) 
Base.convert(::Type{PeriodicFunctionMatrix}, ahr::HarmonicArray)  where T = 
         PeriodicFunctionMatrix(t::Real -> hreval(ahr,t), ahr.period)
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
   PeriodicSymbolicMatrix(A.f(t), A.period)
end
Base.convert(::Type{PeriodicSymbolicMatrix}, ahr::HarmonicArray)  where T = 
   PeriodicSymbolicMatrix(hr2psm(ahr), ahr.period)
# function PeriodicSymbolicMatrix(ahr::HarmonicArray, period::Real = ahr.period)
#    @variables t
#    PeriodicSymbolicMatrix(hreval(ahr,t)[1], period)
# end


# conversion to continuous-time HarmonicArray
Base.convert(::Type{HarmonicArray}, A::PeriodicTimeSeriesMatrix) where T = ts2hr(A)


# conversions to continuous-time PeriodicTimeSeriesMatrix
function PeriodicTimeSeriesMatrix(A::PeriodicMatrix{:d,T}, period::Real = A.period) where {T <: Real}
   N = length(A.M) 
   N <= 1 && (return PeriodicTimeSeriesMatrix(A.M, period))
   m, n = size(A.M[1])
   (any(size.(A.M,1) .!= m) || any(size.(A.M,2) .!= n)) && 
         error("only periodic matrices with constant dimensions supported")
   PeriodicTimeSeriesMatrix(A.M, period)
end
PeriodicTimeSeriesMatrix(A::PeriodicArray{:d,T}, period::Real = A.period) where {T <: Real} = 
   PeriodicTimeSeriesMatrix([A.M[:,:,i] for i in 1:size(A.M,3)], period)

