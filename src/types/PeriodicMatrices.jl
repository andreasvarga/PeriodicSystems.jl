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
It is assumed that `M[i] := M[mod(i-1,p)+1]` for arbitrary `i`. 
All component matrices are allowed to have arbitrary (time-varying) dimensions.
The component matrices `M`, the period `T`, the number of subperiods `k`, the discrete period `p` 
and the sample time `Δ` can be accessed via `A.M`, `A.period`, `A.nperiod`, `A.dperiod` and `A.Ts`, respectively. 
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
set_period(A::AbstractVecOrMat,period::Real) = A
function set_period(A::AbstractPeriodicArray{:d,T}, period::Real) where T
   period > 0 || error("period must be positive") 
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicMatrix{:d,T}(A.M, Aperiod*d, A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicMatrix{:d,T}(A.M, Aperiod/n, nperiod)
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
isconstant(A::PeriodicMatrix) = (A.dperiod == 1) || all([A.M[1] == A.M[i] for i in 2:A.dperiod])
Base.size(A::PeriodicMatrix) = (size.(A.M,1),size.(A.M,2))
Base.size(A::PeriodicMatrix, d::Integer) = size.(A.M,d)
Base.length(A::PeriodicMatrix) = A.dperiod
Base.eltype(A::PeriodicMatrix{:d,T}) where T = T

function Base.getindex(A::PM, ind::Int) where PM <: PeriodicMatrix
   A.M[mod(ind-1,A.dperiod)+1]
end
function Base.lastindex(A::PM) where PM <: PeriodicMatrix
   return A.dperiod
end
function Base.getindex(A::PM, inds...) where PM <: PeriodicMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicMatrix{:d,eltype(A)}([A.M[i][rows,cols] for i in 1:length(A)], A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicMatrix
   return length(A) > 0 ? minimum(lastindex.(A.M,dim)) : 0
end


"""
    SwitchingPeriodicMatrix(M, ns, T; nperiod = k) -> A::SwitchingPeriodicMatrix

Discrete-time switching periodic matrix representation.

The discrete-time switching periodic matrix object `A` is built from a 
`p`-vector `M` of real matrices, a `p`-vector `ns` of increasing positive integers representing the
discrete switching moments, 
the associated time period `T` and 
the number of subperiods specified via the keyword argument `nperiod = k`. 

`M` contains the component matrices `M[i]`, `i = 1,..., p`, which defines 
a sequence of `N := ns[p]` of matrices `S[1], ..., S[N]`, 
such that `S[j] = M[i]` for `j ∈ [ns[i-1]+1, ..., ns[i]]` with `ns[0] := 0`.
`S[j]` is the `j`-th value `A(Δ(j-1))` of a time periodic matrix `A(t)`
of subperiod `T′ := T/k`, with `Δ := T′/N`, the associated sample time. 
All component matrices are allowed to have arbitrary (time-varying) dimensions.
The component matrices `M`, the integer vector `ns`, the period `T`, 
the number of subperiods `k`, the discrete period `N` 
and the sample time `Δ` can be accessed via `A.M`, `A.ns`, `A.period`, `A.nperiod`, `A.dperiod` and `A.Ts`, respectively. 

The j-th time value `A(Δ(j-1))` can be determined as `A[j]`. 
It is assumed that `A[j] := A[mod(j-1,N)+1]` for arbitrary `j`. 
"""
struct SwitchingPeriodicMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   M::Vector{Array{T,2}}
   ns::Vector{Int}
   period::Float64
   nperiod::Int
end

# additional constructors
#function  SwitchingPeriodicMatrix(M::Vector{<: Matrix}, ns::Vector{Int}, period::Real; nperiod::Int = 1) 
function  SwitchingPeriodicMatrix(M::Vector{MT}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {MT <: Array} 
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   any(ndims.(M) .> 2) && error("only vectors with vector or matrix elements supported")
   p = length(ns)
   p == length(M) || error("number of component matrices must be equal to the dimension of ns") 
   ns[1] > 0 || error("ns must have only strictly increasing positive values")
   for i in 1:p-1
       ns[i+1] > ns[i] || error("ns must have only strictly increasing positive values")
   end
   m = size.(M,2)
   T = promote_type(eltype.(M)...)
   return any(m .== 1) ?  SwitchingPeriodicMatrix{:d,T}([T.(reshape(M[i],size(M[i],1),m[i])) for i in 1:p], ns, Float64(period), nperiod)  :  
                   SwitchingPeriodicMatrix{:d,T}([T.(M[i]) for i in 1:p], ns, Float64(period), nperiod) 
end
SwitchingPeriodicMatrix{:d,T}(A::Vector{Matrix{T}}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {T} = 
   SwitchingPeriodicMatrix{:d,T}(A, ns, period, nperiod)
# SwitchingPeriodicMatrix(M::Vector{Matrix{T}}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {T <: Real} = 
#    SwitchingPeriodicMatrix{:d,T}(M, ns, period, nperiod)
SwitchingPeriodicMatrix(M::VecOrMat{T}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {T <: Real} =
   SwitchingPeriodicMatrix{:d,T}([reshape(M,size(M,1),size(M,2))], ns, period, nperiod)
SwitchingPeriodicMatrix(M::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real} =
   SwitchingPeriodicMatrix{:d,T}([reshape(M,size(M,1),size(M,2))], [nperiod], period, 1)

# period change
function SwitchingPeriodicMatrix{:d,T}(A::SwitchingPeriodicMatrix{:d,T1}, period::Real) where {T,T1}
   period > 0 || error("period must be positive") 
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      SwitchingPeriodicMatrix{:d,T}([T.(A.M[i]) for i in 1:length(A.M)], A.ns, Aperiod*d, A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      SwitchingPeriodicMatrix{:d,T}([T.(A.M[i]) for i in 1:length(A.M)], A.ns, Aperiod/n, nperiod)
   end
end


function Base.getproperty(A::SwitchingPeriodicMatrix, d::Symbol)  
   if d === :dperiod
      return getfield(A, :ns)[end]
   elseif d === :Ts
      return A.period/A.dperiod/A.nperiod
   else
      getfield(A, d)
   end
end
Base.propertynames(A::SwitchingPeriodicMatrix) = (:dperiod, :Ts, fieldnames(typeof(A))...)
isconstant(A::SwitchingPeriodicMatrix) = (length(A.ns) == 1) || all([A.M[1] == A.M[i] for i in 2:length(A.ns)])
Base.size(A::SwitchingPeriodicMatrix) = (size.(A.M,1),size.(A.M,2))
Base.size(A::SwitchingPeriodicMatrix, d::Integer) = size.(A.M,d)
Base.length(A::SwitchingPeriodicMatrix) = length(A.ns)
Base.eltype(A::SwitchingPeriodicMatrix{:d,T}) where T = T

function Base.getindex(A::SPM, ind::Int) where SPM <: SwitchingPeriodicMatrix
   A.M[findfirst(A.ns .>= mod(ind-1,A.dperiod)+1)]
end
function Base.lastindex(A::SPM) where SPM <: SwitchingPeriodicMatrix
   return A.dperiod
end
function Base.getindex(A::PM, inds...) where PM <: SwitchingPeriodicMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   SwitchingPeriodicMatrix{:d,eltype(A)}([A.M[i][rows,cols] for i in 1:length(A.M)], A.ns, A.period, A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: SwitchingPeriodicMatrix
   return length(A) > 0 ? minimum(lastindex.(A.M,dim)) : 0
end


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
The component matrices `M`, the period `T`, the number of subperiods `k`, the discrete period `p` 
and the sample time `Δ` can be accessed via `A.M`, `A.period`, `A.nperiod`, `A.dperiod` and `A.Ts`, respectively. 
"""
struct PeriodicArray{Domain,T} <: AbstractPeriodicArray{Domain,T}
    M::Array{T,3}
    period::Float64
    nperiod::Int
end 
# additional constructors
function  PeriodicArray{:d,T}(M::AbstractArray{T,3}, period::Real; nperiod::Int = 1) where {T <: Real} 
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
PeriodicArray(M::AbstractArray{T,3}, period::Real; nperiod::Int = 1) where {T <: Real} = PeriodicArray{:d,T}(M, period, nperiod)
PeriodicArray(M::AbstractVecOrMat{T}, period::Real; nperiod::Int = 1) where T = PeriodicArray(reshape(M,size(M,1),size(M,2),1), period; nperiod)
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
isconstant(A::PeriodicArray) = (A.dperiod == 1) || all([view(A.M,:,:,1) == view(A.M,:,:,i) for i in 2:A.dperiod])
Base.size(A::PeriodicArray) = (size(A.M,1),size(A.M,2))
Base.size(A::PeriodicArray, d::Integer) = size(A.M,d)
Base.length(A::PeriodicArray) = A.dperiod
Base.eltype(A::PeriodicArray{:d,T}) where T = T

function Base.getindex(A::PM, ind::Int) where PM <: PeriodicArray
   A.M[:,:,mod(ind-1,A.dperiod)+1]
end
function Base.lastindex(A::PM) where PM <: PeriodicArray
   return A.dperiod
end
function Base.getindex(A::PM, inds...) where PM <: PeriodicArray
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicArray{:d,eltype(A)}(A.M[rows,cols,:], A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicArray
   return lastindex(A.M,dim) 
end

iscontinuous(A::AbstractPeriodicArray) = typeof(A).parameters[1] == :c 
#iscontinuous(A) = typeof(A).parameters[1] == :c 
iscontinuous(A::Type) = A.parameters[1] == :c 


"""
    SwitchingPeriodicArray(M, ns, T; nperiod = k) -> A::SwitchingPeriodicArray

Discrete-time switching periodic array representation.

The discrete-time switching periodic array object `A` is built from a `m×n×p` real array
`M`, a `p`-vector `ns` of increasing positive integers representing the
discrete switching moments, the associated time period `T` and 
the number of subperiods specified via the keyword argument `nperiod = k`. 

`M` contains the cyclic component matrices `M[:,:,i]`, `i = 1,..., p`, which defines 
a sequence of `N := ns[p]` of matrices `S[1], ..., S[N]`, 
such that `S[j] = `M[:,:,i]` for `j ∈ [ns[i-1]+1, ..., ns[i]]` with `ns[0] := 0`.
`S[j]` is the `j`-th value `A(Δ(j-1))` of a time periodic matrix `A(t)`
of subperiod `T′ := T/k`, with `Δ := T′/N`, the associated sample time. 
The component matrices `M`, the integer vector `ns`, the period `T`, 
the number of subperiods `k`, the discrete period `N` 
and the sample time `Δ` can be accessed via `A.M`, `A.ns`, `A.period`, `A.nperiod`, `A.dperiod` and `A.Ts`, respectively. 

The j-th time value `A(Δ(j-1))` can be determined as `A[j]`. 
It is assumed that `A[j] := A[mod(j-1,N)+1]` for arbitrary `j`. 
"""
struct SwitchingPeriodicArray{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   M::Array{T,3}
   ns::Vector{Int}
   period::Float64
   nperiod::Int
end

# additional constructors
function  SwitchingPeriodicArray{:d,T}(M::Array{T,3}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {T <: Real} 
   period > 0 || error("period must be positive")       
   nperiod > 0 || error("number of subperiods must be positive") 
   p = length(ns)
   p == size(M,3) || error("number of component matrices must be equal to the dimension of ns") 
   ns[1] > 0 || error("ns must have only strictly increasing positive values")
   for i in 1:p-1
       ns[i+1] > ns[i] || error("ns must have only strictly increasing positive values")
   end
   SwitchingPeriodicArray{:d,T}(M, ns, Float64(period), nperiod) 
end
function SwitchingPeriodicArray{:d,T}(A::SwitchingPeriodicArray{:d,T1}, period::Real) where {T,T1}
   period > 0 || error("period must be positive") 
   #isconstant(A) && (return PeriodicArray{:d,T}(convert(Array{T,3},A.M), period; nperiod = 1))
   Aperiod = A.period

   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      SwitchingPeriodicArray{:d,T}(convert(Array{T,3},A.M), A.ns, Aperiod*d; nperiod = A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      SwitchingPeriodicArray{:d,T}(convert(Array{T,3},A.M), A.ns, Aperiod/n; nperiod)
   end
end

SwitchingPeriodicArray(M::Array{T,3}, ns::Vector{Int}, period::Real; nperiod::Int = 1) where {T <: Real} = SwitchingPeriodicArray{:d,T}(M, ns, period; nperiod)
SwitchingPeriodicArray(M::VecOrMat{T}, period::Real; nperiod::Int = 1) where T = SwitchingPeriodicArray(reshape(M,size(M,1),size(M,2),1), [1], period; nperiod)

function Base.getproperty(A::SwitchingPeriodicArray, d::Symbol)  
   if d === :dperiod
      return getfield(A, :ns)[end]
   elseif d === :Ts
      return A.period/A.dperiod/A.nperiod
   else
      getfield(A, d)
   end
end
Base.propertynames(A::SwitchingPeriodicArray) = (:dperiod, :Ts, fieldnames(typeof(A))...)
isconstant(A::SwitchingPeriodicArray) = (length(A.ns) == 1) || all([view(A.M,:,:,1) == view(A.M,:,:,i) for i in 2:length(A.ns)])
Base.size(A::SwitchingPeriodicArray) = (size(A.M,1),size(A.M,2))
Base.size(A::SwitchingPeriodicArray, d::Integer) = size(A.M,d)
Base.length(A::SwitchingPeriodicArray) = length(A.ns)
Base.eltype(A::SwitchingPeriodicArray{:d,T}) where {T} = T

function Base.getindex(A::SPM, ind::Int) where SPM <: SwitchingPeriodicArray
   A.M[:,:,findfirst(A.ns .>= mod(ind-1,A.dperiod)+1)]
end
function Base.lastindex(A::SPM) where SPM <: SwitchingPeriodicArray
   return A.dperiod
end
function Base.getindex(A::PM, inds...) where PM <: SwitchingPeriodicArray
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic array")
   rows, cols = index2range(inds...) 
   SwitchingPeriodicArray{:d,eltype(A)}(A.M[rows,cols,:], A.ns, A.period, A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: SwitchingPeriodicArray
   return lastindex(A.M,dim) 
end



"""
    PeriodicFunctionMatrix(f, T; nperiod = k) -> A::PeriodicFunctionMatrix

Continuous-time periodic function matrix representation.

The continuous-time periodic function matrix object `A` is built from a 
time periodic real matrix function `f(t)` of real time variable `t`, 
the associated time period `T` and the associated number of subperiods
specified via the keyword argument `nperiod = k`. 
It is assumed that  `f(t) = f(t+T/k)` for any real time value `t`.
The function `f(t)`, the period `T`, the row and column dimensions 
of `f(t)`, the number of subperiods `k` can be accessed via `A.f`, `A.period`, 
the tuple `A.dims` and `A.nperiod`, respectively. 
"""
struct PeriodicFunctionMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T}
   f::Function
   period::Float64
   dims::Tuple{Int,Int}
   nperiod::Int
   _isconstant::Bool
   function PeriodicFunctionMatrix{Domain,T}(f::Function, period::Real, dims::Tuple{Int,Int}, nperiod::Int, isconst::Bool) where {T, Domain}
      period > 0 || error("period must be positive") 
      nperiod > 0 || error("number of subperiods must be positive") 
      #isperiodic(f,period/nperiod) || error("non-periodic function matrix")
      isc = PeriodicSystems.isconstant(f,period/nperiod)
      # if isconst 
      #    isc == isconst || @warn "non-constant function matrix detected: isconst has been reset to false"
      # else
      #    isc == isconst || @warn "constant function matrix detected: isconst has been reset to true"
      # end
      F0 = f(period)
      nd = ndims(F0)
      nd <= 2 || error("two-dimensional function array expected, got an $nd -dimensional array")
      m, n = size(F0,1),size(F0,2)
      dims == (m, n) || error("expected dimensions $((m,n)), got $dims")  
      new{:c,T}(f,period,dims,nperiod,isc)
   end
end 

# additional constructors
function PeriodicFunctionMatrix{:c,Tf}(f::Function, period::Real; isconst::Bool = false, nperiod::Int = 1) where {Tf} 
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   F0 = f(period)
   if typeof(F0) <: Real 
      return eltype(F0) == Tf ? PeriodicFunctionMatrix{:c,Tf}(t -> [f(t)], Float64(period), (1,1), nperiod, isconst) :
                                PeriodicFunctionMatrix{:c,Tf}(t -> [Tf(f(Tf(t)))], Float64(period), (1,1), nperiod, isconst)
   end
   nd = ndims(F0)
   nd <= 2 || error("two-dimensional function array expected, got an $nd -dimensional array")
   m, n = size(F0,1),size(F0,2)
   eltype(F0) == Tf ? PeriodicFunctionMatrix{:c,Tf}(t -> n == 1 ? reshape(f(t),m,n) : f(t), Float64(period), (m,n), nperiod, isconst) :
                      PeriodicFunctionMatrix{:c,Tf}(t -> n == 1 ? convert(Matrix{Tf},reshape(f(Tf(t)),m,n)) : convert(Matrix{Tf},f(Tf(t))), Float64(period), (m,n), nperiod, isconst)
end
PeriodicFunctionMatrix(f::F, period::Real; isconst::Bool = false, nperiod::Int = 1) where {F<:Function}  = 
             PeriodicFunctionMatrix{:c,eltype(f(period))}(f, period; isconst, nperiod)
function PeriodicFunctionMatrix(A::VecOrMat{T}, period::Real) where {T <: Real}
   if T == Num
      @variables t
      f = eval(build_function(reshape(A,size(A,1),size(A,2)), t, expression=Val{false})[1])
      PeriodicFunctionMatrix{:c,Float64}(t -> f(t), period; isconst = false)
   else
      PeriodicFunctionMatrix{:c,T}(t -> reshape(A,size(A,1),size(A,2)), period; isconst = true)
   end
end
PeriodicFunctionMatrix{:c,Float64}(A::VecOrMat{T}, period::Real) where {T <: Real} = PeriodicFunctionMatrix(Float64.(A), period)

function PeriodicFunctionMatrix{:c,T}(at::PeriodicFunctionMatrix, period::Real) where {T}
   period > 0 || error("period must be positive") 
   Aperiod = at.period
   Aperiod == period && T == eltype(at) && (return at) 
   isconstant(at) && (return PeriodicFunctionMatrix{:c,T}(at.f, period, at.dims, 1, true))
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicFunctionMatrix{:c,T}(at.f, Aperiod*d, at.dims, at.nperiod*d, false)
   elseif period < Aperiod
      nperiod = div(at.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicFunctionMatrix{:c,T}(at.f, Aperiod/n, at.dims, at.nperiod, false)
   end
end
PeriodicFunctionMatrix(at::PeriodicFunctionMatrix, period::Real) = PeriodicFunctionMatrix{:c,eltype(at)}(at, period) 
#set_period(A::PeriodicFunctionMatrix, period::Real) = PeriodicFunctionMatrix{:c,eltype(A)}(A,period)

# function PeriodicFunctionMatrix(at::PeriodicFunctionMatrix, period::Real = at.period; isconst::Bool = isconstant(at))
#    # at.period = period
#    # at._isconstant = isconst
#    # return at
#    return PeriodicFunctionMatrix(at.f, period; isconst)
# end
# properties
isconstant(A::PeriodicFunctionMatrix) = A._isconstant || PeriodicSystems.isconstant(A.f,A.period/A.nperiod)

# function isperiodic(f::Function, period::Real)  
#    t = rand(typeof(period))*period
#    return f(t) ≈ f(t+period) || isapprox(f(t),f(t+period),rtol = 100*sqrt(eps(period)))
# end
function PeriodicSystems.isconstant(f::Function, period::Real; rtol::Float64 = eps(), atol::Float64 = sqrt(eps()))  
   return abs(optimize(t->-norm(f(t)-f(0),Inf),0,period,Optim.Brent(),rel_tol = rtol).minimum) < atol 
end

#isperiodic(A::PeriodicFunctionMatrix) = isconstant(A) ? true : isperiodic(A.f,A.period/A.nperiod)
Base.size(A::PeriodicFunctionMatrix) = A.dims
Base.size(A::PeriodicFunctionMatrix, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(A::PeriodicFunctionMatrix{:c,T}) where T = T
function Base.getindex(A::PM, inds...) where PM <: PeriodicFunctionMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicFunctionMatrix{:c,eltype(A)}(t->A.f(t)[rows,cols], A.period; isconst = A._isconstant, nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicFunctionMatrix
   lastindex(A.f(0),dim)
end

index2range(ind1, ind2) = (index2range(ind1), index2range(ind2))
index2range(ind::T) where {T<:Number} = ind:ind
index2range(ind::T) where {T<:AbstractArray} = ind
index2range(ind::Colon) = ind

"""
    PeriodicSymbolicMatrix(F, T; nperiod = k) -> A::PeriodicSymbolicMatrix

Continuous-time periodic symbolic matrix representation.
 
The continuous-time periodic symbolic matrix object `A` is built from `F`, a 
symbolic periodic real matrix or vector of symbolic variable `t`, 
the associated time period `T` and the associated number of subperiods
specified via the keyword argument `nperiod = k`. 
It is assumed that  `F(t) = F(t+T/k)` for any real time value `t`.
The symbolic matrix `F`, the period `T` and the number of subperiods `k` 
can be accessed via `A.F`, `A.period` and `A.nperiod`, respectively.
"""
struct PeriodicSymbolicMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   F::Matrix{<:Num}
   period::Float64
   nperiod::Int
end 
# additional constructors
function  PeriodicSymbolicMatrix{:c,T}(F::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Num} 
   period > 0 || error("period must be positive")       
   nperiod > 0 || error("number of subperiods must be positive") 
   # check that array F is depending only on t
   tt = rand()
   @variables t
   Ft = substitute.(F, (Dict(t => tt),))
   m, n = size(Ft,1), size(Ft,2)
   any(length.(Symbolics.get_variables.(Ft)) .> 0 ) && error("t must be the only variable in F")
   PeriodicSymbolicMatrix{:c,T}(n == 1 ? reshape(F,m,n) : F, Float64(period), nperiod) 
end
PeriodicSymbolicMatrix(F::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Union{Num,Real}} = 
    PeriodicSymbolicMatrix{:c,Num}(Num.(F), period; nperiod)
# PeriodicSymbolicMatrix(F::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real} = 
#     PeriodicSymbolicMatrix{:c,Num}(Num.(F), Float64(period), nperiod)
# PeriodicSymbolicMatrix{:c,Num}(F::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Number} = 
#     PeriodicSymbolicMatrix{:c,Num}(Num.(F), Float64(period), nperiod)
function PeriodicSymbolicMatrix{:c,T}(A::PeriodicSymbolicMatrix, period::Real) where {T}
   period > 0 || error("period must be positive") 
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicSymbolicMatrix{:c,T}(A.F, Aperiod*d, A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicSymbolicMatrix{:c,T}(A.F, Aperiod/n, A.nperiod)
   end
end
set_period(A::PM, period::Real) where {T, PM <: AbstractPeriodicArray{:c,T}} = PM(A,period)

# properties 
isconstant(A::PeriodicSymbolicMatrix) = all(length.(Symbolics.get_variables.(A.F)) .== 0)
# function isperiodic(A::VecOrMat{T}, period::Real) where {T <: Num} 
#    tt = rand()
#    @variables t
#    At = substitute.(A, (Dict(t => tt),))
#    return norm(At - substitute.(A, (Dict(t => tt+period),))) <= eps(10.)*max(1.,norm(At)) 
# end
#isperiodic(A::PeriodicSymbolicMatrix) = isconstant(A) ? true : isperiodic(A.F,A.period)
Base.size(A::PeriodicSymbolicMatrix) = size(A.F)
Base.size(A::AbstractPeriodicArray, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(A::PeriodicSymbolicMatrix{:c,T}) where T = T

function Base.getindex(A::PM, inds...) where PM <: PeriodicSymbolicMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicSymbolicMatrix{:c,eltype(A)}(A.F[rows,cols], A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicSymbolicMatrix
   lastindex(A.F,dim)
end


struct FourierFunctionMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   M::Fun
   period::Float64
   nperiod::Int
end
# additional constructors
"""
     FourierFunctionMatrix(Afun, T) -> A::FourierFunctionMatrix

Continuous-time Fourier function matrix representation.

The Fourier function matrix object `A` of period `T` is built using
the Fourier series representation of a periodic matrix `Afun(t)` of subperiod `T′ = T/k`, 
where each entry of `Afun(t)` has the form

             p
      a_0 +  ∑ ( ac_i*cos(i*t*2*π/T′)+as_i*sin(i*2*π*t/T′) ) ,
            i=1 

where `k ≥ 1` is the number of subperiods (default: `k = 1`).   
The matrix `Afun` containing the Fourier representation, the period `T` and the 
number of subperiods `k` can be accessed via `A.M`, `A.period` and `A.nperiod`, respectively.
"""
function FourierFunctionMatrix{:c,T}(A::Fun, period::Real) where {T}
   period > 0 || error("period must be positive") 
   n, m = size(A,1), size(A,2)
   sint = domain(A)
   (sint.a == 0 && sint.b > 0) || error("the domain must be of the form 0..period")
   ti = rationalize(period/sint.b)
   denominator(ti) == 1 || error("only integer multiple periods supported")
   FourierFunctionMatrix{:c,eltype(domain(A))}(m == 1 ? reshape(A,n,m) : A, Float64(period), numerator(ti)) 
end
FourierFunctionMatrix(A::Fun, period::Real; nperiod::Int = 1)  = 
       FourierFunctionMatrix{:c,eltype(domain(A))}(A, period, nperiod) 
# FourierFunctionMatrix(A::Matrix{<:Fun}, period::Real; nperiod::Int = 1)  = 
#        FourierFunctionMatrix{:c,eltype(domain(A))}(A, period, nperiod) 

FourierFunctionMatrix(A::Fun)  = FourierFunctionMatrix{:c,eltype(domain(A))}(A::Fun, domain(A).b) 
function FourierFunctionMatrix{:c,T}(A::FourierFunctionMatrix, period::Real) where {T}
   period > 0 || error("period must be positive") 
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      FourierFunctionMatrix{:c,T}(A.M, Aperiod*d, A.nperiod*d)
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      FourierFunctionMatrix{:c,T}(A.M, Aperiod/n, A.nperiod)
   end
end
FourierFunctionMatrix(A0::VecOrMat{T}, period::Real) where {T <: Real}  = 
    FourierFunctionMatrix{:c,float(T)}(Fun(t->float(T).(A0),Fourier(0..period)), period) 
FourierFunctionMatrix{:c,T}(A0::VecOrMat, period::Real) where {T <: Real}  = 
    FourierFunctionMatrix{:c,float(T)}(Fun(t->float(T).(A0),Fourier(0..period)), period) 
function isconstant(A::FourierFunctionMatrix)
   k = length(size(A.M))
   if  k > 1
     for i = 1:size(A,1)
          for j = 1: size(A,2)
              ncoefficients(chop(A.M[i,j])) <= 1 || (return false)
          end
      end
   elseif k == 1
      for i = 1:size(A,1)
          ncoefficients(chop(A.M[i])) <= 1 || (return false)
      end
   else
      ncoefficients(chop(A.M)) <= 1 || (return false)      
   end
   return true
end
#isperiodic(A::FourierFunctionMatrix) = true
Base.size(A::FourierFunctionMatrix) = (size(A.M,1),size(A.M,2))
Base.eltype(A::FourierFunctionMatrix{:c,T}) where T = T

function Base.getindex(A::PM, inds...) where PM <: FourierFunctionMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   FourierFunctionMatrix{:c,eltype(A)}(A.M[rows,cols], A.period, A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: FourierFunctionMatrix
   size(A.M,dim)
end


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
function HarmonicArray{:c,T}(Ahr::Array{<:Complex,3}, period::Real; nperiod::Int = 1) where {T}
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   (size(Ahr,3) > 0 && iszero(imag(view(Ahr,:,:,1)))) || error("imaginary part of constant term must be zero")
   HarmonicArray{:c,T}(convert(Array{Complex{T},3},Ahr), Float64(period), nperiod) 
end
HarmonicArray(Ahr::Array{Complex{T},3}, period::Real; nperiod::Int = 1) where T = HarmonicArray{:c,T}(Ahr, period; nperiod)
# change period and type
function HarmonicArray{:c,T}(A::HA, period::Real) where {HA <: HarmonicArray} where {T}
   period > 0 || error("period must be positive") 
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
                               Asin::Union{Vector{MT},Nothing}, period::Real; nperiod::Int = 1) where {T <: Real, MT <: VecOrMat{T}}
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
   HarmonicArray{:c,T}(ahr, period, nperiod)
end
HarmonicArray(A0::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real}  = 
          HarmonicArray(complex(reshape(A0,size(A0,1),size(A0,2),1)), period; nperiod) 
HarmonicArray(A0::VecOrMat{T}, Acos::Vector{MT}, period::Real; nperiod::Int = 1) where {T <: Real, MT <: VecOrMat{T}}  = 
          HarmonicArray(A0, Acos, nothing, period; nperiod) 
HarmonicArray{:c,Float64}(A::VecOrMat{T}, period::Real; nperiod::Int = 1) where {T <: Real} = HarmonicArray(Float64.(A), period; nperiod)


# properties
isconstant(A::HarmonicArray) = size(A.values,3) <= 1 || iszero(view(A.values,:,:,2:size(A.values,3)))
#isperiodic(A::HarmonicArray) = true
Base.size(A::HarmonicArray) = (size(A.values,1),size(A.values,2))
Base.eltype(A::HarmonicArray{:c,T}) where T = T

function Base.getindex(A::PM, inds...) where PM <: HarmonicArray
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   HarmonicArray{:c,eltype(A)}(A.values[rows,cols,:], A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: HarmonicArray
   lastindex(A.values,dim)
end
"""
    PeriodicSwitchingMatrix(At, ts, T; nperiod = k) -> A::PeriodicSwitchingMatrix

Continuous-time periodic switching matrix representation.

The continuous-time periodic switching matrix object `A` of period `T` is built from a 
`p`-vector `At` of real matrices, a `p`-vector `ts` of increasingly ordered switching time values with `ts[1] = 0`, and 
the associated subperiod `T′ = T/k`, where `k ≥ 1` is the number of subperiods (default: `k = 1`). 
`At` contains the cyclic component matrices `At[i]`, `i = 1,..., p`, 
where `At[i]` is the constant value of a time periodic matrix `A(t)` of period `T′`
for `t ∈ [ts[i],ts[i+1])`, if `i < p`, or `t ∈ [ts[i],T′)`, if `i = p`. 
It is assumed that `At[i] := At[mod(i-1,p)+1]` and `ts[i] := ts[mod(i-1,p)+1]` for arbitrary `i`. 
All component matrices must have the same dimensions.
The component matrices `At`, the switching times `ts`, the period `T` and the number of subperiods `k`
can be accessed via `A.values`, `A.ts`, `A.period`, and `A.nperiod`, respectively. 
"""
struct PeriodicSwitchingMatrix{Domain,T} <: AbstractPeriodicArray{Domain,T} 
   values::Vector{Array{T,2}}
   ts::Vector{Float64}
   period::Float64
   nperiod::Int
end
# additional constructors
function PeriodicSwitchingMatrix{:c,T}(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, ts::Vector{T1}, period::Real; nperiod::Int = 1) where {T <: Real, T1 <: Real} 
   period > 0 || error("period must be positive") 
   nperiod > 0 || error("number of subperiods must be positive") 
   N = length(At) 
   N == length(ts) || error("number of time values must be equal to the number of matrix values")
   #N <= 1 && (return PeriodicTimeSeriesMatrix{:c,T}(At, Float64(period); nperiod) ) # the constant matrix case 
   n1, n2 = size(At[1],1), size(At[1],2)
   (all(size.(At,1) .== n1) && all(size.(At,2) .== n2)) || error("all component matrices must have the same dimensions")
   (N < 2 || all(diff(ts) .> 0)) || error("time values must be strictly increasing")
   ts[1] == 0 || error("first time value must be zero")
   ts[N] < period/nperiod || error("last time value must be less than the sub-period")
   # adjust final data to matrix type
   PeriodicSwitchingMatrix{:c,T}(n2 == 1 ? [reshape(At[j],n1,n2) for j in 1:N] : At, Float64.(ts), Float64(period), nperiod) 
end
PeriodicSwitchingMatrix(At::Union{Vector{Vector{T}},Vector{Matrix{T}}}, ts::Vector{T1}, period::Real; nperiod::Int = 1) where {T <: Real, T1 <: Real}  = 
     PeriodicSwitchingMatrix{:c,T}(At, ts, period; nperiod)  
PeriodicSwitchingMatrix{:c,T}(A::Union{Vector{Vector{T1}},Vector{Matrix{T1}}}, ts::Vector{T2}, period::Real; nperiod::Int = 1) where {T<: Real, T1 <: Real,T2 <: Real} = 
     PeriodicSwitchingMatrix([T.(A[i]) for i in 1:length(A)], ts, period; nperiod)
function PeriodicSwitchingMatrix(At::VecOrMat{T}, period::Real; ts::Vector{T1} = [0.0], nperiod::Int = 1) where {T <: Real, T1 <: Real}
   PeriodicSwitchingMatrix([reshape(At,size(At,1),size(At,2))], ts, period; nperiod)
end
function PeriodicSwitchingMatrix{:c,T}(A::PeriodicSwitchingMatrix{:c,T1}, period::Real) where {T<: Real, T1 <: Real}
   Aperiod = A.period
   r = rationalize(Aperiod/period)
   n, d = numerator(r), denominator(r)
   min(n,d) == 1 || error("new period is incommensurate with the old period")
   if period >= Aperiod
      PeriodicSwitchingMatrix{:c,T}([T.(A.values[i]) for i in 1:length(A)], A.ts, Aperiod*d; nperiod = A.nperiod*d)      
   elseif period < Aperiod
      nperiod = div(A.nperiod,n)
      nperiod < 1 && error("new period is incommensurate with the old period")
      PeriodicSwitchingMatrix{:c,T}([T.(A.values[i]) for i in 1:length(A)], A.ts, Aperiod/n; nperiod)
   end
end
function  PeriodicSwitchingMatrix(M::AbstractArray{T,3}, ts::Vector{T1}, period::Real; nperiod::Int = 1) where {T <: Real, T1 <: Real} 
   period > 0 || error("period must be positive")       
   nperiod > 0 || error("number of subperiods must be positive") 
   p = length(ts)
   size(M,3) == p || error("number of component matrices must be equal to the dimension of ts") 
   ts[1] == 0 || error("ts must have the first value equal to zero")
   for i in 1:p-1
       ts[i+1] > ts[i] || error("ts must have only strictly increasing positive values")
   end
   PeriodicSwitchingMatrix{:c,T}([M[:,:,i] for i in 1:p], ts, Float64(period), nperiod) 
end


# properties
isconstant(At::PeriodicSwitchingMatrix) = length(At.values) <= 1
#isperiodic(At::PeriodicSwitchingMatrix) = true
Base.length(At::PeriodicSwitchingMatrix) = length(At.values) 
Base.size(At::PeriodicSwitchingMatrix) = length(At) > 0 ? size(At.values[1]) : (0,0)
Base.eltype(At::PeriodicSwitchingMatrix{:c,T}) where {T} = T

function Base.getindex(A::PM, inds...) where PM <: PeriodicSwitchingMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicSwitchingMatrix{:c,eltype(A)}([A.values[i][rows,cols] for i in 1:length(A)], A.ts, A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicSwitchingMatrix
   return length(A) > 0 ? lastindex(A.values[1],dim) : 0
end


"""
    PeriodicTimeSeriesMatrix(At, T; nperiod = k) -> A::PeriodicTimeSeriesMatrix

Continuous-time periodic time series matrix representation.

The continuous-time periodic time series matrix object `A` of period `T` is built from a 
`p`-vector `At` of real matrices and the associated subperiod `T′ = T/k`, where
`k ≥ 1` is the number of subperiods (default: `k = 1`). 
`At` contains the cyclic component matrices `At[i]`, `i = 1,..., p`, 
where `At[i]` represents the value `A(Δ*(i-1))` of a time periodic matrix `A(t)`
of period `T′`, with `Δ := T′/p`, the associated sampling time.
It is assumed that `At[i] := At[mod(i-1,p)+1]` for arbitrary `i`. 
All component matrices must have the same dimensions.
The component matrices `At`, the period `T` and the number of subperiods `k`
can be accessed via `A.values`, `A.period`, and `A.nperiod`, respectively. 
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
#isperiodic(At::PeriodicTimeSeriesMatrix) = true
Base.length(At::PeriodicTimeSeriesMatrix) = length(At.values) 
Base.size(At::PeriodicTimeSeriesMatrix) = length(At) > 0 ? size(At.values[1]) : (0,0)
Base.eltype(At::PeriodicTimeSeriesMatrix{:c,T}) where T = T

function Base.getproperty(A::PeriodicTimeSeriesMatrix, d::Symbol)  
   if d === :ts
      ns = length(A)
      return collect((0:ns-1)*(A.period/A.nperiod/ns))
   else
      getfield(A, d)
   end
end

function Base.getindex(A::PTS, ind::Int) where PTS <: PeriodicTimeSeriesMatrix
   A.values[mod(ind-1,length(A.values))+1]
end
function Base.lastindex(A::PTS) where PTS <: PeriodicTimeSeriesMatrix
   return length(A.values)
end

function Base.getindex(A::PM, inds...) where PM <: PeriodicTimeSeriesMatrix
   size(inds, 1) != 2 &&
       error("Must specify 2 indices to index a periodic matrix")
   rows, cols = index2range(inds...) 
   PeriodicTimeSeriesMatrix{:c,eltype(A)}([A.values[i][rows,cols] for i in 1:length(A)], A.period; nperiod = A.nperiod)
end
function Base.lastindex(A::PM, dim::Int) where PM <: PeriodicTimeSeriesMatrix
   return length(A) > 0 ? lastindex(A.values[1],dim) : 0
end


# conversions to discrete-time PeriodicMatrix
Base.convert(::Type{<:PeriodicMatrix}, A::PeriodicArray{:d,T}) where T = 
             PeriodicMatrix{:d,T}([A.M[:,:,i] for i in 1:size(A.M,3)],A.period; nperiod = A.nperiod)
Base.convert(::Type{<:PeriodicMatrix}, A::SwitchingPeriodicMatrix{:d,T}) where {T} = 
             PeriodicMatrix{:d,T}([A[i] for i in 1:A.dperiod],A.period; nperiod = A.nperiod)
Base.convert(::Type{<:PeriodicMatrix}, A::SwitchingPeriodicArray{:d,T}) where {T} = 
             PeriodicMatrix{:d,T}([A[i] for i in 1:A.dperiod],A.period; nperiod = A.nperiod)
function Base.convert(::Type{<:PeriodicArray}, A::SwitchingPeriodicArray{:d,T}) where {T} 
   X = Array{T,3}(undef, size(A.M,1), size(A.M,2), A.dperiod)
   for i in 1:A.dperiod
       copyto!(view(X,:,:,i), A[i])
   end 
   PeriodicArray{:d,T}(X, A.period; nperiod = A.nperiod)
end
function Base.convert(::Type{SwitchingPeriodicMatrix}, A::PeriodicMatrix{:d,T}) where {T}
   ns = Int[]
   na = length(A.M)
   k = 0
   i = 1
   while i <= na
       i == na && (push!(ns,na); break)
       for j = i+1:na
           k += 1
           !isequal(A.M[i],A.M[j]) && (push!(ns,k); break)
       end
       i = k+1
   end
   SwitchingPeriodicMatrix{:d,T}([A.M[ns[i]] for i in 1:length(ns)],ns, A.period; nperiod = A.nperiod)
end
Base.convert(::Type{PeriodicArray}, A::PM) where {PM <: PeriodicMatrix} = pm2pa(A)
Base.convert(::Type{PeriodicArray{:d,T}}, A::PM) where {T, PM <: PeriodicMatrix} = pm2pa(A)
function Base.convert(::Type{SwitchingPeriodicMatrix}, A::PeriodicArray{:d,T}) where {T}
   ns = Int[]
   na = size(A.M,3)
   k = 0
   i = 1
   while i <= na
       i == na && (push!(ns,na); break)
       for j = i+1:na
           k += 1
           !isequal(A.M[:,:,i],A.M[:,:,j]) && (push!(ns,k); break)
       end
       i = k+1
   end
   SwitchingPeriodicMatrix{:d,T}([A.M[:,:,ns[i]] for i in 1:length(ns)],ns, A.period; nperiod = A.nperiod)
end
Base.convert(::Type{PeriodicArray}, A::SwitchingPeriodicMatrix) = convert(PeriodicArray,convert(PeriodicMatrix,A))
#Base.convert(::Type{PeriodicArray}, A::SwitchingPeriodicArray) = convert(PeriodicArray,convert(PeriodicMatrix,A))
function Base.convert(::Type{SwitchingPeriodicMatrix}, A::SwitchingPeriodicArray{:d,T}) where {T}
   SwitchingPeriodicMatrix{:d,T}([A.M[:,:,A.ns[i]] for i in 1:length(A.ns)], A.ns, A.period; nperiod = A.nperiod)
end
function Base.convert(::Type{SwitchingPeriodicArray}, A::PeriodicArray{:d,T}) where {T}
   ns = Int[]
   na = size(A.M,3)
   k = 0
   i = 1
   while i <= na
       i == na && (push!(ns,na); break)
       for j = i+1:na
           k += 1
           !isequal(A.M[:,:,i],A.M[:,:,j]) && (push!(ns,k); break)
       end
       i = k+1
   end
   nx = length(ns)
   X = Array{T,3}(undef, size(A.M,1), size(A.M,2), nx)
   for i = 1:nx
       copyto!(view(X,:,:,i), view(A.M,:,:,ns[i]))
   end
   SwitchingPeriodicArray{:d,T}(X, ns, A.period; nperiod = A.nperiod)
end



# conversions to continuous-time PeriodicFunctionMatrix
function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicFunctionMatrix) where T
   return eltype(A) == T ? A : PeriodicFunctionMatrix{:c,T}(x -> T.(A.f(T(x))), A.period, A.dims, A.nperiod, A._isconstant)
end

function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::FourierFunctionMatrix) where T
   return eltype(A) == T ? A : PeriodicFunctionMatrix{:c,T}(x -> T.(A.M(T(x))), A.period, size(A), A.nperiod, isconstant(A))
end
function Base.convert(::Type{PeriodicFunctionMatrix}, A::FourierFunctionMatrix)
   return PeriodicFunctionMatrix{:c,eltype(A)}(x -> A.M(x), A.period, size(A.M), A.nperiod, isconstant(A))
end

function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicSymbolicMatrix) where T
   @variables t
   f = eval(build_function(A.F, t, expression=Val{false}, nanmath=false)[1])
   PeriodicFunctionMatrix{:c,T}(x -> f(T(x)), A.period, size(A), A.nperiod, isconstant(A))
end
function Base.convert(::Type{PeriodicFunctionMatrix}, A::PeriodicSymbolicMatrix) 
   @variables t
   f = eval(build_function(A.F, t, expression=Val{false}, nanmath=false)[1])
   return PeriodicFunctionMatrix{:c,Float64}(x -> f(x), A.period, size(A), A.nperiod, isconstant(A))
end
# function Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicSymbolicMatrix) where T
#    @variables t
#    f = eval(build_function(A.F, t, expression=Val{false})[1])
#    PeriodicFunctionMatrix{:c,T}(x -> f(x), A.period, size(A), A.nperiod, isconstant(A))
# end

# PeriodicFunctionMatrix(ahr::HarmonicArray, period::Real = ahr.period; exact = true)  = 
#           PeriodicFunctionMatrix(t::Real -> hreval(ahr,t;exact)[1], period) 
Base.convert(::Type{PeriodicFunctionMatrix}, ahr::HarmonicArray)  = 
         PeriodicFunctionMatrix{:c,real(eltype(ahr.values))}(t::Real -> hreval(ahr,t), ahr.period, size(ahr), ahr.nperiod, isconstant(ahr))
Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, ahr::HarmonicArray)  where T = 
         PeriodicFunctionMatrix{:c,T}(t::Real -> hreval(ahr,T(t)), ahr.period, size(ahr), ahr.nperiod, isconstant(ahr))
#Base.convert(::Type{PeriodicFunctionMatrix}, At::PeriodicTimeSeriesMatrix) = ts2pfm(At; method = "cubic")
Base.convert(::Type{PeriodicFunctionMatrix}, At::PeriodicTimeSeriesMatrix; method = "cubic") = ts2pfm(At; method)
#Base.convert(::Type{PeriodicFunctionMatrix}, At::PeriodicSwitchingMatrix; method = "cubic") = ts2pfm(convert(PeriodicTimeSeriesMatrix,At); method)
#Base.convert(::Type{PeriodicFunctionMatrix}, At::PeriodicSwitchingMatrix; method = "linear") = tsw2pfm(At; method)
Base.convert(::Type{PeriodicFunctionMatrix}, A::PeriodicSwitchingMatrix) = 
         PeriodicFunctionMatrix{:c,eltype(A)}(t::Real -> tpmeval(A,t), A.period, size(A), A.nperiod, isconstant(A))
Base.convert(::Type{PeriodicFunctionMatrix{:c,T}}, A::PeriodicSwitchingMatrix) where {T} =
         PeriodicFunctionMatrix{:c,T}(t::Real -> tpmeval(A,t), A.period, size(A), A.nperiod, isconstant(A))


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

# conversions to continuous-time Fourier function matrix
function Base.convert(::Type{FourierFunctionMatrix}, A::PeriodicFunctionMatrix) 
   T = float(eltype(A))
   return FourierFunctionMatrix{:c,T}(Fun(x -> T.(A.f(x)), Fourier(0..A.period/A.nperiod)), Float64(A.period), A.nperiod)
end
function Base.convert(::Type{FourierFunctionMatrix}, A::HarmonicArray) 
   tA = convert(PeriodicFunctionMatrix,A)
   return FourierFunctionMatrix{:c,eltype(tA)}(Fun(x -> tA.f(x), Fourier(0..tA.period/tA.nperiod)), Float64(tA.period), tA.nperiod)
end
function Base.convert(::Type{FourierFunctionMatrix}, A::PeriodicSymbolicMatrix) 
   tA = convert(PeriodicFunctionMatrix,A)
   return FourierFunctionMatrix{:c,eltype(tA)}(Fun(x -> tA.f(x), Fourier(0..tA.period/tA.nperiod)), Float64(tA.period), tA.nperiod)
end
function Base.convert(::Type{FourierFunctionMatrix}, A::PeriodicTimeSeriesMatrix) 
   tA = convert(PeriodicFunctionMatrix,A)
   return FourierFunctionMatrix{:c,eltype(tA)}(Fun(x -> tA.f(x), Fourier(0..tA.period/tA.nperiod)), Float64(tA.period), tA.nperiod)
end

# conversions to continuous-time PeriodicSymbolicMatrix
function Base.convert(::Type{PeriodicSymbolicMatrix}, A::PeriodicFunctionMatrix) 
   @variables t
   # PeriodicSymbolicMatrix(Num.(A.f(t)), A.period; nperiod = A.nperiod)
   M = try 
      Num.(A.f(t))
   catch 
      return convert(PeriodicSymbolicMatrix,convert(HarmonicArray,A))
   end
   PeriodicSymbolicMatrix(M, A.period; nperiod = A.nperiod)
end
function Base.convert(::Type{PeriodicSymbolicMatrix{:c,T}}, A::PeriodicFunctionMatrix) where {T}
   try 
      @variables t
      PeriodicSymbolicMatrix(Num.(A.f(t)), A.period; nperiod = A.nperiod)
   catch
      convert(PeriodicSymbolicMatrix,convert(HarmonicArray,A))
   end
end
Base.convert(::Type{PeriodicSymbolicMatrix}, ahr::HarmonicArray) = 
   PeriodicSymbolicMatrix(hr2psm(ahr), ahr.period; nperiod = ahr.nperiod)
Base.convert(::Type{PeriodicSymbolicMatrix}, A::PeriodicTimeSeriesMatrix) = 
   PeriodicSymbolicMatrix(hr2psm(convert(HarmonicArray,A)), A.period; nperiod = A.nperiod)
Base.convert(::Type{PeriodicSymbolicMatrix}, A::FourierFunctionMatrix) =
   convert(PeriodicSymbolicMatrix,convert(HarmonicArray,A))


# conversion to continuous-time HarmonicArray
Base.convert(::Type{HarmonicArray}, A::PeriodicTimeSeriesMatrix) = ts2hr(A)
Base.convert(::Type{HarmonicArray}, A::PeriodicFunctionMatrix) = pfm2hr(A)
Base.convert(::Type{HarmonicArray{:c, T}}, A::PeriodicFunctionMatrix) where {T} = pfm2hr(A)
Base.convert(::Type{HarmonicArray}, A::PeriodicSymbolicMatrix) = psm2hr(A)
#Base.convert(::Type{HarmonicArray}, A::FourierFunctionMatrix) = pfm2hr(convert(PeriodicFunctionMatrix,A))
Base.convert(::Type{HarmonicArray}, A::FourierFunctionMatrix) = ffm2hr(A)
Base.convert(::Type{HarmonicArray}, A::PeriodicSwitchingMatrix) = ts2hr(convert(PeriodicTimeSeriesMatrix,A;ns=128))

# conversions to PeriodicTimeSeriesMatrix
function Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicSymbolicMatrix; ns::Int = 128)
   ns > 0 || throw(ArgumentError("number of samples must be positive, got $ns"))
   tA = convert(PeriodicFunctionMatrix,A)
   PeriodicTimeSeriesMatrix(tA.f.((0:ns-1)*tA.period/ns/tA.nperiod), tA.period; nperiod = tA.nperiod)
end
function Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicFunctionMatrix; ns::Int = 128)
   ns > 0 || throw(ArgumentError("number of samples must be positive, got $ns"))
   isconstant(A) ? PeriodicTimeSeriesMatrix(A.f.(0), A.period) : 
                   PeriodicTimeSeriesMatrix(A.f.((0:ns-1)*A.period/ns/A.nperiod), A.period; nperiod = A.nperiod)
end
function Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::HarmonicArray; ns::Int = 128)
   ns > 0 || throw(ArgumentError("number of samples must be positive, got $ns"))
   PeriodicTimeSeriesMatrix(tvmeval(A, collect((0:ns-1)*A.period/ns/A.nperiod)), A.period; nperiod = A.nperiod)
end
Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicMatrix) =
    convert(PeriodicTimeSeriesMatrix, pm2pa(A))
Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::PeriodicArray) =
    PeriodicTimeSeriesMatrix([A.M[:,:,i] for i in 1:size(A.M,3)], A.period; nperiod = A.nperiod)
function Base.convert(::Type{PeriodicTimeSeriesMatrix}, A::FourierFunctionMatrix; ns::Int = 128)
    convert(PeriodicTimeSeriesMatrix,convert(PeriodicFunctionMatrix,A);ns)
end
function Base.convert(::Type{<:PeriodicTimeSeriesMatrix}, A::PM; ns::Int = 1) where {PM <: PeriodicSwitchingMatrix}
   ns > 0 || throw(ArgumentError("number of samples must be positive, got $ns"))
   isconstant(A) && (return PeriodicTimeSeriesMatrix(A.values[1], A.period; nperiod = A.nperiod))
   dt = [diff(A.ts); A.period/A.nperiod-A.ts[end]]
   Δ = minimum(dt)
   nlim = 2^10
   for a in dt
      a ≈ Δ && continue
      den = rationalize(a/Δ).den
      if den > nlim 
         ns = max(128,length(A.ts))
         @warn "incommensurate sampling times: ns = $ns is used"
         #ns = 1024; 
         Δ = A.period/ns/A.nperiod
         return PeriodicTimeSeriesMatrix([tpmeval(A,(i-1)*Δ) for i in 1:ns], A.period; nperiod = A.nperiod)
      end
      den == 1 || (Δ /= den)
   end
   #K = rationalize(A.period/A.nperiod/Δ).num # minimum number of possible sample times
   K = round(Int,A.period/A.nperiod/Δ) # minimum number of possible sample times
   if ns > 1 
      K = lcm(K,ns)
      Δ = A.period/K/A.nperiod
   end
   return PeriodicTimeSeriesMatrix([tpmeval(A,(i-1)*Δ) for i in 1:K], A.period; nperiod = A.nperiod)
end

# conversions to PeriodicSwitchingMatrix
function Base.convert(::Type{PeriodicSwitchingMatrix}, A::PeriodicFunctionMatrix; ns::Int = 128)
   ns > 0 || throw(ArgumentError("number of samples must be positive, got $ns"))
   isconstant(A) && (return PeriodicSwitchingMatrix(A(0), A.period))
   convert(PeriodicSwitchingMatrix,convert(PeriodicTimeSeriesMatrix,A;ns))
end

function Base.convert(::Type{PeriodicSwitchingMatrix}, A::PeriodicTimeSeriesMatrix) 
   ns = Int[]
   na = length(A.values)
   k = 0
   i = 1
   while i <= na
       i == na && (push!(ns,na); break)
       for j = i+1:na
           k += 1
           !isequal(A.values[i],A.values[j]) && (push!(ns,k); break)
       end
       i = k+1
   end
   Δ = A.period/na/A.nperiod
   PeriodicSwitchingMatrix{:c,eltype(A)}([A.values[ns[i]] for i in 1:length(ns)], [0; Δ*ns[1:end-1]], A.period; nperiod = A.nperiod)
end
