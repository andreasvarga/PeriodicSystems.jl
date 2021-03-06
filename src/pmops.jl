# Operations with periodic arrays
function pmshift(A::PeriodicArray, k::Int = 1)
    return PeriodicArray(A.M[:,:,mod.(k:A.dperiod+k-1,A.dperiod).+1], A.period; nperiod = A.nperiod)
end
function LinearAlgebra.transpose(A::PeriodicArray)
    return PeriodicArray(permutedims(A.M,(2,1,3)), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicArray)
    return PeriodicArray(permutedims(A.M,(2,1,3)), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.norm(A::PeriodicArray, p::Real = 2)
    return norm([norm(A.M[:,:,i],p) for i in 1:A.dperiod],p)
end
function +(A::PeriodicArray, B::PeriodicArray)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    m, n, pa = size(A.M)
    mb, nb, pb = size(B.M)
    (m, n) == (mb, nb) || throw(DimensionMismatch("A and B must have the same dimensions"))
    nta = numerator(rationalize(period/A.period))
    ntb = numerator(rationalize(period/B.period))
    pa == pb && nta == 1 && ntb == 1 && (return PeriodicArray(A.M+B.M, A.period; nperiod = A.nperiod))
    K = nta*A.nperiod*pa
    p = lcm(pa,pb)
    T = promote_type(eltype(A),eltype(B))
    X = Array{T,3}(undef, m, n, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        X[:,:,i] = A.M[:,:,ia]+B.M[:,:,ib]
    end
    return PeriodicArray(X, period; nperiod = div(K,p))
end
+(A::PeriodicArray, C::AbstractMatrix) = +(A, PeriodicArray(C, A.Ts; nperiod = 1))
+(A::AbstractMatrix, C::PeriodicArray) = +(PeriodicArray(A, C.Ts; nperiod = 1), C)
-(A::PeriodicArray) = PeriodicArray(-A.M, A.period; nperiod = A.nperiod)
-(A::PeriodicArray, B::PeriodicArray) = +(A,-B)
function *(A::PeriodicArray, B::PeriodicArray)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    m, na, pa = size(A.M)
    mb, n, pb = size(B.M)
    na == mb || throw(DimensionMismatch("A and B have incompatible dimension"))
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    #K == B.nperiod*pb || error("A and B must have the same sampling time")
    #pa == pb && (return PeriodicArray(A.M*B.M, A.period; nperiod = A.nperiod))
    p = lcm(pa,pb)
    T = promote_type(eltype(A),eltype(B))
    X = Array{T,3}(undef, m, n, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        mul!(view(X,:,:,i), view(A.M,:,:,ia), view(B.M,:,:,ib))
    end
    return PeriodicArray(X, period; nperiod = div(K,p))
end
*(A::PeriodicArray, C::AbstractMatrix) = *(A, PeriodicArray(C, A.Ts; nperiod = 1))
*(A::AbstractMatrix, C::PeriodicArray) = *(PeriodicArray(A, C.Ts; nperiod = 1), C)
*(A::PeriodicArray, C::Real) = PeriodicArray(C*A.M, A.period; nperiod = A.nperiod)
*(A::Real, C::PeriodicArray) = PeriodicArray(A*C.M, C.period; nperiod = C.nperiod)

LinearAlgebra.issymmetric(A::PeriodicArray) = all([issymmetric(A.M[:,:,i]) for i in 1:A.dperiod])

# Operations with periodic matrices
function pmshift(A::PeriodicMatrix, k::Int = 1)
    return PeriodicMatrix(A.M[mod.(k:A.dperiod+k-1,A.dperiod).+1], A.period; nperiod = A.nperiod)
end
function LinearAlgebra.transpose(A::PeriodicMatrix)
    return PeriodicMatrix(copy.(transpose.(A.M)), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicMatrix)
    return PeriodicMatrix(copy.(adjoint.(A.M)), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.norm(A::PeriodicMatrix, p::Real = 2)
    return norm(norm.(A.M, p) ,p)
end
function +(A::PeriodicMatrix, B::PeriodicMatrix)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    pa = length(A) 
    pb = length(B)
    p = lcm(pa,pb)
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    T = promote_type(eltype(A),eltype(B))
    X = Vector{Matrix{T}}(undef, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        size(A.M[ia]) == size(B.M[ib]) || throw(DimensionMismatch("A and B have incompatible dimension"))
        X[i] = A.M[ia]+B.M[ib]
    end
    return PeriodicMatrix(X, period; nperiod = div(K,p))
end
+(A::PeriodicMatrix, C::AbstractMatrix) = +(A, PeriodicMatrix(C, A.Ts; nperiod = 1))
+(A::AbstractMatrix, C::PeriodicMatrix) = +(PeriodicMatrix(A, C.Ts; nperiod = 1), C)
-(A::PeriodicMatrix) = PeriodicMatrix(-A.M, A.period; nperiod = A.nperiod)
-(A::PeriodicMatrix, B::PeriodicMatrix) = +(A,-B)
function *(A::PeriodicMatrix, B::PeriodicMatrix)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    pa = length(A) 
    pb = length(B)
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    p = lcm(pa,pb)
    T = promote_type(eltype(A),eltype(B))
    X = Vector{Matrix{T}}(undef, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        size(A.M[ia],2) == size(B.M[ib],1) || throw(DimensionMismatch("A and B have incompatible dimension"))
        X[i] = A.M[ia]*B.M[ib]
    end
    return PeriodicMatrix(X, period; nperiod = div(K,p))
end
*(A::PeriodicMatrix, C::AbstractMatrix) = *(A, PeriodicMatrix(C, A.Ts; nperiod = 1))
*(A::AbstractMatrix, C::PeriodicMatrix) = *(PeriodicMatrix(A, C.Ts; nperiod = 1), C)
*(A::PeriodicMatrix, C::Real) = PeriodicMatrix(C.*A.M, A.period; nperiod = A.nperiod)
*(A::Real, C::PeriodicMatrix) = PeriodicMatrix(A.*C.M, C.period; nperiod = C.nperiod)

LinearAlgebra.issymmetric(A::PeriodicMatrix) = all([issymmetric(A.M[i]) for i in 1:A.dperiod])
  

# Operations with periodic function matrices
function derivative(A::PeriodicFunctionMatrix{:c,T};  h = A.period*sqrt(eps(T)), method = "cd") where {T}
    isconstant(A) && (return PeriodicFunctionMatrix{:c,T}(t -> zeros(T,A.dims...), A.period, A.dims, A.nperiod, true))
    method == "cd" ? (return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t-h))/(2*h), A.period, A.dims, A.nperiod, false)) :
                     (return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t))/h, A.period, A.dims, A.nperiod, false))
end
function LinearAlgebra.transpose(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> transpose(A.f(t)), A.period, (A.dims[2],A.dims[1]), A.nperiod, A._isconstant)
end
function LinearAlgebra.adjoint(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> adjoint(A.f(t)), A.period, (A.dims[2],A.dims[1]), A.nperiod, A._isconstant)
end
function norm(A::PeriodicFunctionMatrix, p::Real = 2; K = 128) 
    nrm = zero(eltype(A))
    Δ = A.period/K
    t = zero(eltype(Δ))
    for i = 1:K
        nrm = max(nrm,opnorm(A.f(t),p))
        t += Δ
    end 
    return nrm
end
function +(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    period = promote_period(A, B)
    A.dims == B.dims || throw(DimensionMismatch("A and B must have the same dimensions"))
    nperiod = gcd(A.nperiod,B.nperiod)
    T = promote_type(eltype(A),eltype(B))
    if isconstant(A) && isconstant(B)
       return PeriodicFunctionMatrix{:c,T}(t -> A.f(0)+B.f(0), period, A.dims, nperiod, true)
    else
       return PeriodicFunctionMatrix{:c,T}(t -> A.f(t)+B.f(t), period, A.dims, nperiod, false)
    end
end
+(A::PeriodicFunctionMatrix, C::AbstractMatrix) = +(A, PeriodicFunctionMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicFunctionMatrix) = +(PeriodicFunctionMatrix(A, C.period), C)
-(A::PeriodicFunctionMatrix) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> -A.f(t), A.period, A.dims, A.nperiod,A._isconstant)
-(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix) = +(A,-B)
(+)(A::PeriodicFunctionMatrix, J::UniformScaling{<:Real}) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> A.f(t)+J, A.period, A.dims, A.nperiod,A._isconstant)
(+)(J::UniformScaling{<:Real}, A::PeriodicFunctionMatrix) = +(A,J)
(-)(A::PeriodicFunctionMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicFunctionMatrix) = +(-A,J)

function *(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    period = promote_period(A, B)
    A.dims[2] == B.dims[1] || throw(DimensionMismatch("A and B have incompatible dimension"))
    nperiod = gcd(A.nperiod,B.nperiod)
    T = promote_type(eltype(A),eltype(B))
    if isconstant(A) && isconstant(B)
        return PeriodicFunctionMatrix{:c,T}(t -> A.f(0)*B.f(0), period, A.dims, nperiod, true)
     else
        return PeriodicFunctionMatrix{:c,T}(t -> A.f(t)*B.f(t), period, A.dims, nperiod, false)
     end
 end
*(A::PeriodicFunctionMatrix, C::AbstractMatrix) = *(A, PeriodicFunctionMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicFunctionMatrix) = *(PeriodicFunctionMatrix(A, C.period), C)
*(A::PeriodicFunctionMatrix, C::Real) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> C.*A.f(t), A.period, A.dims, A.nperiod,A._isconstant)
*(A::Real, C::PeriodicFunctionMatrix) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> A.*C.f(t), C.period, C.dims, C.nperiod,C._isconstant)

# Operations with periodic symbolic matrices
function derivative(A::PeriodicSymbolicMatrix) 
    @variables t   
    return PeriodicSymbolicMatrix{:c,Num}(Symbolics.derivative(A.F,t), A.period, nperiod = A.nperiod)
end
function LinearAlgebra.transpose(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(transpose(A.F)), A.period, nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(adjoint(A.F)), A.period, nperiod = A.nperiod)
end
function norm(A::PeriodicSymbolicMatrix, p::Real = 2; K = 128) 
    @variables t   
    nrm = 0. 
    Δ = A.period/K
    ts = zero(eltype(Δ))
    for i = 1:K       
        nrm = max(nrm,opnorm(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))),p) )
        ts += Δ
    end 
    return nrm
end
function +(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    return PeriodicSymbolicMatrix{:c,Num}(A.F + B.F, period; nperiod)
end
+(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = +(A, PeriodicSymbolicMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = +(PeriodicSymbolicMatrix(A, C.period), C)
-(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(-A.F, A.period; nperiod = A.nperiod)
-(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix) = +(A,-B)
function (+)(A::PeriodicSymbolicMatrix, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    PeriodicSymbolicMatrix(A.F+Matrix(J(n)), A.period; nperiod = A.nperiod)
end
(+)(J::UniformScaling{<:Real}, A::PeriodicSymbolicMatrix) = +(A,J)
(-)(A::PeriodicSymbolicMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicSymbolicMatrix) = +(-A,J)

function *(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    return PeriodicSymbolicMatrix{:c,Num}(A.F * B.F, period; nperiod)
 end
*(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = *(A, PeriodicSymbolicMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = *(PeriodicSymbolicMatrix(A, C.period), C)
*(A::PeriodicSymbolicMatrix, C::Real) = PeriodicSymbolicMatrix(C*A.F, A.period; nperiod = A.nperiod)
*(C::Real, A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(C*A.F, A.period; nperiod = A.nperiod)

# Operations with harmonic arrays
function derivative(A::HarmonicArray{:c,T}) where {T}
    m, n, l = size(A.values)
    isconstant(A) && (return HarmonicArray{:c,T}(zeros(Complex{T}, m, n, 1), A.period, A.nperiod)) 
    Ahr = similar(A.values)
    ω = 2*pi*A.nperiod/A.period
    Ahr[:,:,1] = zeros(T, m, n)
    for i = 1:l-1
        Ahr[:,:,i+1] .= complex.(imag(A.values[:,:,i+1])*(i*ω),real(A.values[:,:,i+1])*(-i*ω)) 
    end
    return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod)
end
function LinearAlgebra.transpose(A::HarmonicArray{:c,T}) where {T}  
    m, n, l = size(A.values)
    Ahr = similar(Array{Complex{T},3},n,m,l)
    for i = 1:l
        Ahr[:,:,i] .= copy(transpose(A.values[:,:,i])) 
    end
    return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod)
end
LinearAlgebra.adjoint(A::HarmonicArray) = transpose(A)
function norm(A::HarmonicArray, p::Real = 2; K = 128) 
    nrm = 0. 
    Δ = A.period/K
    ts = zero(eltype(Δ))
    for i = 1:K       
        nrm = max(nrm,opnorm(tpmeval(A, ts),p)) 
        ts += Δ
    end 
    return nrm
end
+(A::HarmonicArray, B::HarmonicArray) = convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) + convert(PeriodicFunctionMatrix,B))
# function +(A::HarmonicArray, B::HarmonicArray)
#     period = promote_period(A, B)
#     nperiod = gcd(A.nperiod,B.nperiod)
#     return HarmonicArray{:c,Num}(A.F + B.F, period; nperiod)
# end
+(A::HarmonicArray, C::AbstractMatrix) = +(A, HarmonicArray(C, A.period))
+(A::AbstractMatrix, C::HarmonicArray) = +(HarmonicArray(A, C.period), C)
-(A::HarmonicArray) = HarmonicArray(-A.values, A.period; nperiod = A.nperiod)
-(A::HarmonicArray, B::HarmonicArray) = +(A,-B)
function (+)(A::HarmonicArray, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(A,J)
(-)(A::HarmonicArray, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(-A,J)

*(A::HarmonicArray, B::HarmonicArray) = convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) * convert(PeriodicFunctionMatrix,B))
# function *(A::HarmonicArray, B::HarmonicArray)
#     period = promote_period(A, B)
#     nperiod = gcd(A.nperiod,B.nperiod)
#     return HarmonicArray{:c,Num}(A.F * B.F, period; nperiod)
#  end
*(A::HarmonicArray, C::AbstractMatrix) = *(A, HarmonicArray(C, A.period))
*(A::AbstractMatrix, C::HarmonicArray) = *(HarmonicArray(A, C.period), C)
*(A::HarmonicArray, C::Real) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)
*(C::Real, A::HarmonicArray) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)

# Operations with Fourier function matrices
derivative(A::FourierFunctionMatrix{:c,T}) where {T} = FourierFunctionMatrix{:c,T}(differentiate(A.M), A.period, A.nperiod)
LinearAlgebra.transpose(A::FourierFunctionMatrix{:c,T}) where {T}  = FourierFunctionMatrix{:c,T}(transpose(A.M), A.period, A.nperiod)
LinearAlgebra.adjoint(A::FourierFunctionMatrix) = transpose(A)
function norm(A::FourierFunctionMatrix, p::Real = 2; K = 128) 
    nrm = 0. 
    Δ = A.period/K
    ts = zero(eltype(Δ))
    for i = 1:K       
        nrm = max(nrm,opnorm(tpmeval(A, ts),p)) 
        ts += Δ
    end 
    return nrm  
end
function +(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(A.M+B.M, A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,convert(PeriodicFunctionMatrix,A) + convert(PeriodicFunctionMatrix,B))
end
+(A::FourierFunctionMatrix, C::AbstractMatrix) = +(A, FourierFunctionMatrix(C, A.period))
+(A::AbstractMatrix, C::FourierFunctionMatrix) = +(FourierFunctionMatrix(A, C.period), C)
-(A::FourierFunctionMatrix) = FourierFunctionMatrix(-A.M, A.period)
-(A::FourierFunctionMatrix, B::FourierFunctionMatrix) = +(A,-B)
function (+)(A::FourierFunctionMatrix, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::FourierFunctionMatrix) = +(A,J)
(-)(A::FourierFunctionMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::FourierFunctionMatrix) = +(-A,J)

function *(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(A.M*B.M, A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,convert(PeriodicFunctionMatrix,A) * convert(PeriodicFunctionMatrix,B))
end
*(A::FourierFunctionMatrix, C::AbstractMatrix) = *(A, FourierFunctionMatrix(C, A.period))
*(A::AbstractMatrix, C::FourierFunctionMatrix) = *(FourierFunctionMatrix(A, C.period), C)
*(A::FourierFunctionMatrix, C::Real) = FourierFunctionMatrix(C*A.M, A.period)
*(C::Real, A::FourierFunctionMatrix) = FourierFunctionMatrix(C*A.M, A.period)

# Operations with periodic time-series matrices
function derivative(A::PeriodicTimeSeriesMatrix{:c,T}) where {T}
    N = length(A)
    #tvmdereval(A, (0:N-1)*A.period/A.nperiod/N)
    PeriodicTimeSeriesMatrix{:c,T}(tvmeval(derivative(convert(HarmonicArray,A)), collect((0:N-1)*A.period/A.nperiod/N)), A.period, A.nperiod)
end
LinearAlgebra.transpose(A::PeriodicTimeSeriesMatrix{:c,T}) where {T} = 
    PeriodicTimeSeriesMatrix{:c,T}([copy(transpose(A.values[i])) for i in 1:length(A)], A.period, A.nperiod)
LinearAlgebra.adjoint(A::PeriodicTimeSeriesMatrix) = transpose(A)
function norm(A::PeriodicTimeSeriesMatrix, p::Real = 2) 
    nrm = 0. 
    for i = 1:length(A)       
        nrm = max(nrm,opnorm(A.values[i],p)) 
    end 
    return nrm  
end
function +(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) &&
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[i] for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([B.values[i]+A.values[1] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[1] for i in 1:length(A)], A.period, A.nperiod))
    convert(PeriodicTimeSeriesMatrix,convert(HarmonicArray,A) + convert(HarmonicArray,B))
end
+(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = +(A, PeriodicTimeSeriesMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = +(PeriodicTimeSeriesMatrix(A, C.period), C)
-(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix{:c,eltype(A)}([-A.values[i] for i in 1:length(A)], A.period, A.nperiod)
-(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix) = +(A,-B)
function (+)(A::PeriodicTimeSeriesMatrix, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::PeriodicTimeSeriesMatrix) = +(A,J)
(-)(A::PeriodicTimeSeriesMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicTimeSeriesMatrix) = +(-A,J)

function *(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) &&
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]*B.values[i] for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[1]*B.values[1] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]*B.values[1] for i in 1:length(A)], A.period, A.nperiod))
    convert(PeriodicTimeSeriesMatrix,convert(HarmonicArray,A) * convert(HarmonicArray,B))
end
*(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = *(A, PeriodicTimeSeriesMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = *(PeriodicTimeSeriesMatrix(A, C.period), C)
*(A::PeriodicTimeSeriesMatrix, C::Real) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([A.values[i]*C for i in 1:length(A)], A.period, A.nperiod)
*(C::Real, A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([C*A.values[i] for i in 1:length(A)], A.period, A.nperiod)
