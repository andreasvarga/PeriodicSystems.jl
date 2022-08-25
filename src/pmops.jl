# Operations with periodic arrays
function pmshift(A::PeriodicArray, k::Int = 1)
    return PeriodicArray(A.M[:,:,mod.(k:A.dperiod+k-1,A.dperiod).+1], A.period; nperiod = A.nperiod)
end
function LinearAlgebra.inv(A::PeriodicArray) 
    x = similar(A.M)
    [x[:,:,i] = inv(A.M[:,:,i]) for i in 1:size(A.M,3)]
    return PeriodicArray(x, A.period; nperiod = A.nperiod)
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
-(A::PeriodicArray, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicArray) = +(A, -C)
function (+)(A::PeriodicArray, J::UniformScaling{<:Real})
    x = similar(A.M)
    [x[:,:,i] = A.M[:,:,i] + J for i in 1:size(A.M,3)]
    return PeriodicArray(x, A.period; nperiod = A.nperiod)
end
(+)(J::UniformScaling{<:Real}, A::PeriodicArray) = +(A,J)
(-)(A::PeriodicArray, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicArray) = +(-A,J)


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
/(A::PeriodicArray, C::Real) = *(A, 1/C)

LinearAlgebra.issymmetric(A::PeriodicArray) = all([issymmetric(A.M[:,:,i]) for i in 1:A.dperiod])
Base.iszero(A::PeriodicArray) = iszero(A.M)
function ==(A::PeriodicArray, B::PeriodicArray)
    isconstant(A) && isconstant(B) && (return isequal(A.M, B.M))
    isequal(A.M, B.M) &&  (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicArray, B::PeriodicArray; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.M, B.M; rtol, atol))
    isapprox(A.M, B.M; rtol, atol) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicArray, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(A.M[:,:,i], J; kwargs...) for i in 1:size(A.M,3)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicArray; kwargs...) = isapprox(A, J; kwargs...)


# Operations with periodic matrices
function pmshift(A::PeriodicMatrix, k::Int = 1)
    return PeriodicMatrix(A.M[mod.(k:A.dperiod+k-1,A.dperiod).+1], A.period; nperiod = A.nperiod)
end
LinearAlgebra.inv(A::PeriodicMatrix) = PeriodicMatrix(inv.(A.M), A.period; nperiod = A.nperiod)
function LinearAlgebra.transpose(A::PeriodicMatrix)
    return PeriodicMatrix(copy.(transpose.(A.M)), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicMatrix)
    return PeriodicMatrix(copy.(transpose.(A.M)), A.period; nperiod = A.nperiod)
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
-(A::PeriodicMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicMatrix) = +(A, -C)
(+)(A::PeriodicMatrix, J::UniformScaling{<:Real}) = PeriodicMatrix([A.M[i] + J for i in 1:length(A.M)], A.period; nperiod = A.nperiod)
(+)(J::UniformScaling{<:Real}, A::PeriodicMatrix) = +(A,J)
(-)(A::PeriodicMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicMatrix) = +(-A,J)

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
/(A::PeriodicMatrix, C::Real) = *(A, 1/C)

LinearAlgebra.issymmetric(A::PeriodicMatrix) = all([issymmetric(A.M[i]) for i in 1:A.dperiod])
Base.iszero(A::PeriodicMatrix) = iszero(A.M)
function ==(A::PeriodicMatrix, B::PeriodicMatrix)
    isconstant(A) && isconstant(B) && (return isequal(A.M, B.M))
    isequal(A.M, B.M) &&  (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicMatrix, B::PeriodicMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.M, B.M; rtol, atol))
    isapprox(A.M, B.M; rtol, atol) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicMatrix, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(A.M[i], J; kwargs...) for i in 1:length(A.M)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicMatrix; kwargs...) = isapprox(A, J; kwargs...)


# Operations with periodic function matrices
function derivative(A::PeriodicFunctionMatrix{:c,T};  h = A.period*sqrt(eps(T)), method = "cd") where {T}
    isconstant(A) && (return PeriodicFunctionMatrix{:c,T}(t -> zeros(T,A.dims...), A.period, A.dims, A.nperiod, true))
    method == "cd" ? (return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t-h))/(2*h), A.period, A.dims, A.nperiod, false)) :
                     (return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t))/h, A.period, A.dims, A.nperiod, false))
end
function LinearAlgebra.inv(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> inv(A.f(t)), A.period, A.dims, A.nperiod, A._isconstant)
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
-(A::PeriodicFunctionMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicFunctionMatrix) = +(A, -C)
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
        return PeriodicFunctionMatrix{:c,T}(t -> A.f(0)*B.f(0), period, (A.dims[1],B.dims[2]), nperiod, true)
     else
        return PeriodicFunctionMatrix{:c,T}(t -> A.f(t)*B.f(t), period, (A.dims[1],B.dims[2]), nperiod, false)
     end
 end
*(A::PeriodicFunctionMatrix, C::AbstractMatrix) = *(A, PeriodicFunctionMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicFunctionMatrix) = *(PeriodicFunctionMatrix(A, C.period), C)
*(A::PeriodicFunctionMatrix, C::Real) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> C.*A.f(t), A.period, A.dims, A.nperiod,A._isconstant)
*(A::Real, C::PeriodicFunctionMatrix) = PeriodicFunctionMatrix{:c,eltype(A)}(t -> A.*C.f(t), C.period, C.dims, C.nperiod,C._isconstant)
/(A::PeriodicFunctionMatrix, C::Real) = *(A, 1/C)

Base.iszero(A::PeriodicFunctionMatrix) = iszero(A.f(rand()*A.period))
LinearAlgebra.issymmetric(A::PeriodicFunctionMatrix) = issymmetric(A.f(rand()*A.period))
function ==(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    isconstant(A) && isconstant(B) && (return isequal(A.f(0), B.f(0)))
    ts = rand()*A.period
    isequal(A.f(ts), B.f(ts)) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix; kwargs...)
    isconstant(A) && isconstant(B) && (return isapprox(A.f(0), B.f(0); kwargs...))
    ts = rand()*A.period
    isapprox(A.f(ts), B.f(ts); kwargs...) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicFunctionMatrix, J::UniformScaling{<:Real}; kwargs...)
    isconstant(A) && (return isapprox(A.f(0), J; kwargs...))
    ts = rand()*A.period
    isapprox(A.f(ts), J; kwargs...) 
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicFunctionMatrix; kwargs...) = isapprox(A, J; kwargs...)

# Operations with periodic symbolic matrices

function derivative(A::PeriodicSymbolicMatrix) 
    @variables t   
    return PeriodicSymbolicMatrix{:c,Num}(Symbolics.derivative(A.F,t), A.period, nperiod = A.nperiod)
end
LinearAlgebra.inv(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(inv(A.F), A.period; nperiod = A.nperiod)
function LinearAlgebra.transpose(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(transpose(A.F)), A.period, nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(adjoint(A.F)), A.period, nperiod = A.nperiod)
end
function Symbolics.simplify(A::PeriodicSymbolicMatrix)
    return PeriodicSymbolicMatrix{:c,Num}(Symbolics.simplify.(A.F), A.period; nperiod = A.nperiod)
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
    #return PeriodicSymbolicMatrix{:c,Num}(Symbolics.simplify.(A.F + B.F), period; nperiod)
    return PeriodicSymbolicMatrix{:c,Num}(A.F + B.F, period; nperiod)
end
+(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = +(A, PeriodicSymbolicMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = +(PeriodicSymbolicMatrix(A, C.period), C)
-(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(-A.F, A.period; nperiod = A.nperiod)
-(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix) = +(A,-B)
-(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = +(A, -C)
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
/(A::PeriodicSymbolicMatrix, C::Real) = *(A, 1/C)

Base.iszero(A::PeriodicSymbolicMatrix) = iszero(A.F)
LinearAlgebra.issymmetric(A::PeriodicSymbolicMatrix) = iszero(A.F-transpose(A.F))
function ==(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    isconstant(A) && isconstant(B) && (return iszero(A.F-B.F))
    iszero(A.F-B.F) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
==(A::PeriodicSymbolicMatrix, J::UniformScaling{<:Real}) = iszero(A.F-J) 
==(J::UniformScaling{<:Real},A::PeriodicSymbolicMatrix) = iszero(A.F-J) 
function Base.isapprox(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    @variables t   
    ts = rand()*A.period
    isconstant(A) && isconstant(B) && (return isapprox(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))), Symbolics.unwrap.(substitute.(B.F, (Dict(t => ts),))); rtol, atol))
    isapprox(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))), Symbolics.unwrap.(substitute.(B.F, (Dict(t => ts),))); rtol, atol) && 
        (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicSymbolicMatrix, J::UniformScaling{<:Real}; kwargs...)
    @variables t   
    ts = rand()*A.period
    isapprox(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))), J; kwargs...) 
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicSymbolicMatrix; kwargs...) = isapprox(A, J; kwargs...)


# Operations with harmonic arrays
function pmrand(::Type{T}, n::Int, m::Int, period::Real = 2*pi; nh::Int = 1) where {T}
    HarmonicArray(rand(T,n,m), [rand(T,n,m) for i in 1:nh], [rand(T,n,m) for i in 1:nh], period) 
end    
pmrand(n::Int, m::Int, period::Real = 2*pi; nh::Int = 1) = pmrand(Float64, n, m, period; nh)
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
function LinearAlgebra.inv(A::HarmonicArray)
    convert(HarmonicArray,inv(convert(PeriodicFunctionMatrix,A)))
    # ts = (0:127)*A.period/A.nperiod/128
    # convert(HarmonicArray,PeriodicTimeSeriesMatrix(inv.(tvmeval(Ah,collect(ts))),2*pi))
    #convert(HarmonicArray,inv(convert(PeriodicFunctionMatrix,A)))
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
function +(A::HarmonicArray, B::HarmonicArray)
    if A.period == B.period && A.nperiod == B.nperiod
       m, n, la = size(A.values)
       mb, nb, lb = size(B.values)
       (m, n) == (mb, nb) || throw(DimensionMismatch("A and B must have the same size"))
       T = promote_type(eltype(A),eltype(B))
       lmax = max(la,lb)
       Ahr = zeros(Complex{T},n,m,lmax)
       if la >= lb
          Ahr = copy(A.values) 
          Ahr[:,:,1:lb] .+= B.values[:,:,1:lb]  
       else
          Ahr = copy(B.values) 
          Ahr[:,:,1:la] .+= A.values[:,:,1:la]  
       end
       tol = 10*eps(T)*max(norm(A.values,Inf),norm(B.values,Inf)) 
       l = lmax
       for i = lmax:-1:2
           norm(Ahr[:,:,i],Inf) > tol && break
           l -= 1
       end
       l < lmax && (Ahr = Ahr[:,:,1:l])
       return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod) 
    else
       convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) + convert(PeriodicFunctionMatrix,B))
    end
end
+(A::HarmonicArray, C::AbstractMatrix) = +(A, HarmonicArray(C, A.period))
+(A::AbstractMatrix, C::HarmonicArray) = +(HarmonicArray(A, C.period), C)
-(A::HarmonicArray) = HarmonicArray(-A.values, A.period; nperiod = A.nperiod)
-(A::HarmonicArray, B::HarmonicArray) = +(A,-B)
-(A::HarmonicArray, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::HarmonicArray) = +(A, -C)
function (+)(A::HarmonicArray, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(A,J)
(-)(A::HarmonicArray, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(-A,J)

*(A::HarmonicArray, B::HarmonicArray) = convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) * convert(PeriodicFunctionMatrix,B))
*(A::HarmonicArray, C::AbstractMatrix) = *(A, HarmonicArray(C, A.period))
*(A::AbstractMatrix, C::HarmonicArray) = *(HarmonicArray(A, C.period), C)
*(A::HarmonicArray, C::Real) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)
*(C::Real, A::HarmonicArray) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)
/(A::HarmonicArray, C::Real) = *(A, 1/C)

Base.iszero(A::HarmonicArray) = all([iszero(A.values[:,:,i]) for i in 1:size(A.values,3)])
LinearAlgebra.issymmetric(A::HarmonicArray) = all([issymmetric(A.values[:,:,i]) for i in 1:size(A.values,3)])
function ==(A::HarmonicArray, B::HarmonicArray)
    isconstant(A) && isconstant(B) && (return isequal(A.values, B.values))
    isequal(A.values, B.values) && A.period*B.nperiod == B.period*A.nperiod
end
function Base.isapprox(A::HarmonicArray, B::HarmonicArray; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.values, B.values; rtol, atol) )
    na = size(A.values,3)
    nb = size(B.values,3)
    if na == nb
       return isapprox(A.values, B.values; rtol, atol) && A.period*B.nperiod == B.period*A.nperiod
    elseif na > nb
        tol = atol+rtol*max(norm(A,1),norm(B,1))
        return all([isapprox(A.values[:,:,i], B.values[:,:,i]; rtol, atol) for i in 1:nb]) && 
              all([norm(A.values[:,:,i],1) < tol for i in nb+1:na])  && A.period*B.nperiod == B.period*A.nperiod
    else
        tol = atol+rtol*max(norm(A,1),norm(B,1))
        return all([isapprox(A.values[:,:,i], B.values[:,:,i]; rtol, atol) for i in 1:na]) && 
              all([norm(B.values[:,:,i],1) < tol for i in na+1:nb])  && A.period*B.nperiod == B.period*A.nperiod
    end
end
function Base.isapprox(A::HarmonicArray, J::UniformScaling{<:Real}; kwargs...)
    isconstant(A) && (return isapprox(tpmeval(A,0), J; kwargs...))
    ts = rand()*A.period
    isapprox(tpmeval(A,ts), J; kwargs...) 
end
Base.isapprox(J::UniformScaling{<:Real}, A::HarmonicArray; kwargs...) = isapprox(A, J; kwargs...)


#FourierFunctionMatrices
derivative(A::FourierFunctionMatrix{:c,T}) where {T} = FourierFunctionMatrix{:c,T}(differentiate(A.M), A.period, A.nperiod)
LinearAlgebra.inv(A::FourierFunctionMatrix) = FourierFunctionMatrix(inv(A.M), A.period; nperiod = A.nperiod)
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
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    domain(A.M) == domain(B.M) && (return FourierFunctionMatrix(A.M+B.M, period; nperiod))
    #A.period == B.period && A.nperiod == B.nperiod 
    #A.period == B.period && (return FourierFunctionMatrix(A.M+B.M, A.period))
    FourierFunctionMatrix(Fun(t-> A.M(t),Fourier(0..period)),period)+FourierFunctionMatrix(Fun(t-> B.M(t),Fourier(0..period)),period)
    #convert(FourierFunctionMatrix,convert(PeriodicFunctionMatrix,A) + convert(PeriodicFunctionMatrix,B))
end
+(A::FourierFunctionMatrix, C::AbstractMatrix) = +(A, FourierFunctionMatrix(C, A.period))
+(A::AbstractMatrix, C::FourierFunctionMatrix) = +(FourierFunctionMatrix(A, C.period), C)
-(A::FourierFunctionMatrix) = FourierFunctionMatrix(-A.M, A.period)
-(A::FourierFunctionMatrix, B::FourierFunctionMatrix) = +(A,-B)
-(A::FourierFunctionMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::FourierFunctionMatrix) = +(A, -C)
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
/(A::FourierFunctionMatrix, C::Real) = *(A, 1/C)

Base.iszero(A::FourierFunctionMatrix) = iszero(A.M)
LinearAlgebra.issymmetric(A::FourierFunctionMatrix) = iszero(A.M-transpose(A.M))
function ==(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    isconstant(A) && isconstant(B) && (return iszero(A.M(0)-B.M(0)))
    t = rationalize(A.period/B.period)
    domain(A.M) == domain(B.M) && iszero(A.M-B.M) && (t.num == 1 || t.den == 1 || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::FourierFunctionMatrix, B::FourierFunctionMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.M(0), B.M(0); rtol, atol))
    ts = rand()*A.period
    isapprox(A.M(ts), B.M(ts); rtol, atol) && 
        (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::FourierFunctionMatrix, J::UniformScaling{<:Real}; kwargs...)
    isconstant(A) && (return isapprox(A.M(0), J; kwargs...))
    ts = rand()*A.period
    isapprox(tpmeval(A,ts), J; kwargs...) 
end
Base.isapprox(J::UniformScaling{<:Real}, A::FourierFunctionMatrix; kwargs...) = isapprox(A, J; kwargs...)


# Operations with periodic time-series matrices
function derivative(A::PeriodicTimeSeriesMatrix{:c,T}) where {T}
    N = length(A)
    #tvmdereval(A, (0:N-1)*A.period/A.nperiod/N)
    PeriodicTimeSeriesMatrix{:c,T}(tvmeval(derivative(convert(HarmonicArray,A)), collect((0:N-1)*A.period/A.nperiod/N)), A.period, A.nperiod)
end
LinearAlgebra.inv(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix(inv.(A.values), A.period; nperiod = A.nperiod)
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
-(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = +(A, -C)

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
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[1]*B.values[i] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]*B.values[1] for i in 1:length(A)], A.period, A.nperiod))
    convert(PeriodicTimeSeriesMatrix,convert(HarmonicArray,A) * convert(HarmonicArray,B))
end
*(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = *(A, PeriodicTimeSeriesMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = *(PeriodicTimeSeriesMatrix(A, C.period), C)
*(A::PeriodicTimeSeriesMatrix, C::Real) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([A.values[i]*C for i in 1:length(A)], A.period, A.nperiod)
*(C::Real, A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([C*A.values[i] for i in 1:length(A)], A.period, A.nperiod)
/(A::PeriodicTimeSeriesMatrix, C::Real) = *(A, 1/C)

Base.iszero(A::PeriodicTimeSeriesMatrix) = iszero(A.values)
LinearAlgebra.issymmetric(A::PeriodicTimeSeriesMatrix) = all(issymmetric.(A.values))
function ==(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    isconstant(A) && isconstant(B) && (return isequal(A.values, B.values))
    isequal(A.values, B.values) &&  (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.values, B.values; rtol, atol))
    isapprox(A.values, B.values; rtol, atol) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod)
end
function Base.isapprox(A::PeriodicTimeSeriesMatrix, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(A.values[i], J; kwargs...) for i in 1:length(A.values)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicTimeSeriesMatrix; kwargs...) = isapprox(A, J; kwargs...)
