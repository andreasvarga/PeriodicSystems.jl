# discrete-time case
"""
    PeriodicMatrix(M, T) -> A::PeriodicMatrix

Discrete-time periodic matrix representation. 

The discrete periodic matrix object `A` is built from a 
`p`-vector `M` of real matrices and the associated time period `T`. 
`M` contains the cyclic component matrices `M[i]`, `i = 1,..., p`, 
where `M[i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that `M[k] := M[mod(k-1,p)+1]` for arbitrary `k`. 
All component matrices are allowed to have arbitrary (time-varying) dimensions.
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

The discrete periodic matrix object `A` is built from a `m×n×p` real array
`M` and the associated time period `T`. 
`M` contains the cyclic component matrices `M[:,:,i]`, `i = 1,..., p`, 
where `M[:,:,i]` represents the value `M(Δ(i-1))` of a time periodic matrix `M(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that  `M[:,:,k] := M[:,:,mod(k-1,p)+1]` for arbitrary `k`. 
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

`A = t -> f(t)` is a time function defining a `m×n` time periodic real matrix `A(t)` of period `T`
such that `A(t) = A(t+T)` for any time value `t`. 
"""
struct PeriodicFunctionMatrix{Domain,T} <: AbstractPeriodicArray{Domain}
   f::Function
   period::T
   function PeriodicFunctionMatrix{:c,Tf}(f::Function, period::Real) where {Tf} 
      period > 0 || error("period must be positive") 
      T = eltype(f(0))
      t = rand(T)
      Ft = f(t)
      nd = ndims(Ft)
      nd == 2 || error("two-dimensional function array expected, got an $nd -dimensional array")
      Ft ≈ f(t+period) || @warn "function f is likely not periodic"
      new{:c,T}(t -> f(t), Float64(period)) 
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
 
`F(t)` is a symbolic periodic real matrix or vector of symbolic variable `t` and period `T` 
such that `F(t) = F(t+T)` for any real time value `t`. 
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
     HarmonicArray(A, T) -> Ahr::HarmonicArray

Continuous-time harmonic array representation.

The harmonic array object `Ahr` is built for 
the harmonic representation of a periodic matrix `A(t)` of period `T` in the form

                   p
     A(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T)+As_i*sin(i*2*π*t/T) ) .
                  i=1 

The `m×n×(p+1)` complex array `A` contains the harmonic components as follows:
`A[:,:,1]` contains the constant term `A_0` (the mean value) and
the real and imaginary parts of `A[:,:,i+1]`  
for `i = 1, ..., p` contain the coefficient matrices `Ac_i` and `As_i`, respectively. 
"""
struct HarmonicArray{Domain,T} <: AbstractPeriodicArray{Domain} 
   values::Array{Complex{T},3}
   period::Float64
   function HarmonicArray{:c,T}(A::Array{Complex{T},3}, period::Real) where T
      period > 0 || error("period must be positive") 
      (size(A,3) > 0 && iszero(imag(view(A,:,:,1)))) || error("imaginary part of constant term must be zero")
      new{:c,T}(A, Float64(period)) 
   end
end
HarmonicArray(A::Array{Complex{T},3}, period::Real) where {T} = HarmonicArray{:c,T}(A, period) 
"""
     HarmonicArray(A0, Ac, As, T) -> Ahr::HarmonicArray

Construct a harmonic array representation from the harmonic components.

The harmonic array object `Ahr` is built for 
the harmonic representation of a periodic matrix `A(t)` of period `T` in the form

                   p
     A(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T)+As_i*sin(i*2*π*t/T) ) ,
                  i=1 

where the constant term `A_0` is contained in the real matrix `A0`, and `Ac` and `As` are
vectors of real matrices such that the `i`-th (cosinus) coefficient matrix 
`Ac_i` is contained in `Ac[i]` and the `i`-th (sinus) coefficient matrix 
`As_i` is contained in `As[i]`. `p` is the maximum of length of the vectors of matrices `Ac` and `As`. 
If the length of `Ac` or `As` is less than `p`, then zero trailing matrices are assumed in the respective matrix. 
All component matrices must have the same dimensions.
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
    PeriodicTimeSeriesMatrix(A, T) -> At::PeriodicTimeSeriesMatrix

Continuous-time periodic time series matrix representation.

The continuous-time periodic time series matrix object `At` is built from a 
`p`-vector `A` of real matrices and the associated time period `T`. 
`A` contains the cyclic component matrices `A[i]`, `i = 1,..., p`, 
where `A[i]` represents the value `A(Δ(i-1))` of a time periodic matrix `A(t)`
of period `T`, with `Δ := T/p`, the associated sampling time.
It is assumed that `A[k] := A[mod(k-1,p)+1]` for arbitrary `k`. 
All component matrices must have the same dimensions.
"""
struct PeriodicTimeSeriesMatrix{Domain,T} <: AbstractPeriodicArray{Domain} 
   values::Vector{Array{T,2}}
   period::Float64
   function PeriodicTimeSeriesMatrix{:c,T}(A::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real) where {T <: Real} 
      period > 0 || error("period must be positive") 
      N = length(A) 
      N <= 1 && (return new{:c,T}(A, Float64(period)) ) # the constant matrix case 
      n1, n2 = size(A[1],1), size(A[1],2)
      (all(size.(A,1) .== n1) && all(size.(A,2) .== n2)) || error("all component matrices must have the same dimensions")

      # adjust final data to matrix type
      n2 == 1 ? At = [reshape(A[j],n1,n2) for j in 1:N] : At = A
      new{:c,T}(At, Float64(period)) 
   end
end
PeriodicTimeSeriesMatrix(A::Union{Vector{Vector{T}},Vector{Matrix{T}}}, period::Real) where {T <: Real, Ts <: Real} = 
    PeriodicTimeSeriesMatrix{:c,T}(A, period)  
PeriodicTimeSeriesMatrix(A::VecOrMat{T}, period::Real) where {T <: Real, Ts <: Real}  = 
        PeriodicTimeSeriesMatrix([reshape(A,size(A,1),size(A,2))], period) 

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

