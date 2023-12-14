# Operations with periodic arrays
function pmzeros(::Type{T},m::Vector{Int},n::Vector{Int}) where {T}
    lm = length(m)
    ln = length(n)
    return [zeros(T,m[mod(i-1,lm)+1], n[mod(i-1,ln)+1]) for i in 1:lcm(lm,ln)]
end
pmzeros(m::Vector{Int},n::Vector{Int}) = pmzeros(Float64,m,n)
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
function Base.reverse(A::PeriodicArray)
    return PeriodicArray(reverse(A.M,dims=3), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.opnorm(A::PeriodicArray, p::Union{Real,Missing} = missing)
    k = size(A.M,3)
    x = Array{eltype(A),3}(undef, 1, 1, k)
    if ismissing(p)
        [x[1,1,i] = norm(view(A.M,:,:,i)) for i in 1:k]  # Frobenius noorm
    else
        [x[1,1,i] = opnorm(view(A.M,:,:,i),p) for i in 1:k] # p-norm
    end
    return PeriodicArray(x,A.period; nperiod = A.nperiod)
end
function LinearAlgebra.norm(A::PeriodicArray, p::Real = 2)
    n = norm([norm(view(A.M,:,:,i)) for i in 1:size(A.M,3)],p)
    if p == 2 
       return n*sqrt(A.nperiod)
    elseif isinf(p) 
       return n
    elseif p == 1 
       return n*A.nperiod
    else 
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end
function LinearAlgebra.tr(A::PeriodicArray)
    p = size(A.M,3)
    x = Array{eltype(A),3}(undef, 1, 1, p)
    [x[1,1,i] = tr(view(A.M,:,:,i)) for i in 1:p]
    return PeriodicArray(x,A.period; nperiod = A.nperiod)
end
function trace(A::PeriodicArray)
    t = zero(eltype(A.M))
    for i in 1:size(A.M,3)
        t += tr(view(A.M,:,:,i))
    end
    return t*A.nperiod
end
function +(A::PeriodicArray, B::PeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    m, n, pa = size(A.M)
    mb, nb, pb = size(B.M)
    (m, n) == (mb, nb) || throw(DimensionMismatch("A and B must have the same dimensions"))
    nta = numerator(rationalize(period/A.period))
    ntb = numerator(rationalize(period/B.period))
    pa == pb && nta == 1 && ntb == 1 && (return PeriodicArray(A.M+B.M, A.period; nperiod = A.nperiod))
    K = max(nta*A.nperiod*pa,ntb*B.nperiod*pb)
    #K = nta*A.nperiod*pa
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
function pmsymadd!(A::PeriodicArray, scal = 1)
    m, n = size(A) 
    m == n || throw(ArgumentError("matrix A must be square"))
    if scal == 1
       for i = 1:length(A) 
           inplace_transpose_add!(view(A.M,:,:,i))
       end
    else
        for i = 1:length(A) 
            inplace_transpose_add!(view(A.M,:,:,i),scal)
        end
    end
    return A
end


function *(A::PeriodicArray, B::PeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    m, na, pa = size(A.M)
    mb, n, pb = size(B.M)
    na == mb || throw(DimensionMismatch("A and B have incompatible dimensions"))
    nta = numerator(rationalize(period/A.period))
    ntb = numerator(rationalize(period/B.period))
    K = max(nta*A.nperiod*pa,ntb*B.nperiod*pb)
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
*(J::UniformScaling{<:Real}, A::PeriodicArray) = J.λ*A
*(A::PeriodicArray, J::UniformScaling{<:Real}) = A*J.λ
for (PMF, MF) in ((:pmmuladdsym, :muladdsym!), (:pmmultraddsym, :multraddsym!), (:pmmuladdtrsym,:muladdtrsym!) )
    @eval begin
        function $PMF(A::PeriodicArray,B::PeriodicArray,C::PeriodicArray, (α,β) = (true, true))
            if isconstant(A)
               isconstant(B) || isconstant(C) || B.Ts ≈ C.Ts || error("B and C must have the same sampling time")
            elseif isconstant(B)
               isconstant(C) || A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
            elseif isconstant(C)
                A.Ts ≈ B.Ts || error("A and B must have the same sampling time") 
            else
                A.Ts ≈ B.Ts ≈ C.Ts || error("A, B and C must have the same sampling time")
            end
            period = promote_period(A, B, C)
            pa = length(A) 
            pb = length(B)
            pc = length(C)
            p = lcm(pa,pb,pc)
            T = promote_type(eltype(A),eltype(B),eltype(C))
            n = size(A,1)
            X = Array{T,3}(undef, n, n, p)
            nta = numerator(rationalize(period/A.period))
            K = nta*A.nperiod*pa
        
            for i = 1:p
                ia = mod(i-1,pa)+1
                ib = mod(i-1,pb)+1
                ic = mod(i-1,pc)+1
                copyto!(view(X,:,:,i), view(A.M,:,:,ia))
                $MF(view(X,:,:,i), view(B.M,:,:,ib), view(C.M,:,:,ic),(α,β))
            end
            return PeriodicArray(X, period; nperiod = div(K,p))
        end
        $PMF(A::PeriodicArray,B::PeriodicArray,C::PeriodicArray, α, β) = $PMF(A, B, C, (α,β))
        function $PMF(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix, (α,β) = (true, true)) 
            T = promote_type(eltype(A),eltype(B),eltype(C))
            $MF(LinearAlgebra.copy_oftype(A,T),B,C,(α,β))
        end
    end
end

for PMF in (:pmmuladdsym, :pmmultraddsym, :pmmuladdtrsym )
    for PM in (:PeriodicArray, :PeriodicMatrix)
        @eval begin
            $PMF(A::$PM,B::AbstractMatrix,C::$PM, (α,β) = (true, true)) = $PMF(A, $PM(B, A.period), C, (α,β))
            $PMF(A::$PM,B::$PM,C::AbstractMatrix, (α,β) = (true, true)) = $PMF(A, B, $PM(C, A.period), (α,β))
            $PMF(A::$PM,B::AbstractMatrix,C::AbstractMatrix, (α,β) = (true, true)) = $PMF(A, $PM(B, A.period), $PM(C, A.period), (α,β))
            $PMF(A::AbstractMatrix,B::$PM,C::$PM, (α,β) = (true, true)) = $PMF($PM(A, B.period), B, C, (α,β))
            $PMF(A::AbstractMatrix,B::AbstractMatrix,C::$PM, (α,β) = (true, true)) = $PMF($PM(A, C.period), $PM(B, C.period), C, (α,β))
            $PMF(A::AbstractMatrix,B::$PM,C::AbstractMatrix, (α,β) = (true, true)) = $PMF($PM(A, B.period), B, $PM(C, B.period), (α,β))
        end
    end
end
    
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


function horzcat(A::PeriodicArray, B::PeriodicArray)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    m, na, pa = size(A.M)
    mb, nb, pb = size(B.M)
    m == mb || throw(DimensionMismatch("A and B have incompatible row dimensions"))
    p = lcm(pa,pb)
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    T = promote_type(eltype(A),eltype(B))
    X = Array{T,3}(undef, m, na+nb, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        X[:,:,i] = [view(A.M,:,:,ia) view(B.M,:,:,ib)]
    end
    return PeriodicArray(X, period; nperiod = div(K,p))
end
Base.hcat(A::PeriodicArray, B::PeriodicArray) = horzcat(A,B)
function vertcat(A::PeriodicArray, B::PeriodicArray)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    ma, n, pa = size(A.M)
    mb, nb, pb = size(B.M)
    n == nb || throw(DimensionMismatch("A and B have incompatible column dimensions"))
    p = lcm(pa,pb)
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    T = promote_type(eltype(A),eltype(B))
    X = Array{T,3}(undef, ma+mb, n, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        X[:,:,i] = [view(A.M,:,:,ia); view(B.M,:,:,ib)]
    end
    return PeriodicArray(X, period; nperiod = div(K,p))
end
Base.vcat(A::PeriodicArray, B::PeriodicArray) = vertcat(A,B)

function blockdiag(A::PeriodicArray, B::PeriodicArray)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    ma, na, pa = size(A.M)
    mb, nb, pb = size(B.M)
    p = lcm(pa,pb)
    nta = numerator(rationalize(period/A.period))
    K = nta*A.nperiod*pa
    T = promote_type(eltype(A),eltype(B))
    X = Array{T,3}(undef, ma+mb, na+nb, p)
    for i = 1:p
        ia = mod(i-1,pa)+1
        ib = mod(i-1,pb)+1
        X[:,:,i] = DescriptorSystems.blockdiag(view(A.M,:,:,ia), view(B.M,:,:,ib))
    end
    return PeriodicArray(X, period; nperiod = div(K,p))
end



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
function Base.reverse(A::PeriodicMatrix)
    return PeriodicMatrix(reverse(A.M), A.period; nperiod = A.nperiod)
end
function LinearAlgebra.tr(A::PeriodicMatrix)
    return PeriodicMatrix([[tr(A.M[i])] for i in 1:length(A)], A.period; nperiod = A.nperiod)
end
function trace(A::PeriodicMatrix)
    t = zero(eltype(A))
    for i in 1:length(A)
        t += tr(A.M[i])
    end
    return t*A.nperiod
end
function LinearAlgebra.opnorm(A::PeriodicMatrix, p::Union{Real,Missing} = missing)
    if ismissing(p)
        return PeriodicMatrix([norm(A.M[i]) for i in 1:length(A)], A.period; nperiod = A.nperiod)
    else
        return PeriodicMatrix([opnorm(A.M[i],p) for i in 1:length(A)], A.period; nperiod = A.nperiod)
    end
end
function LinearAlgebra.norm(A::PeriodicMatrix, p::Real = 2)
    n = norm([norm(A.M[i]) for i in 1:length(A)],p)
    if p == 2 
       return n*sqrt(A.nperiod)
    elseif isinf(p) 
       return n
    elseif p == 1 
       return n*A.nperiod
    else 
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end
function +(A::PeriodicMatrix, B::PeriodicMatrix)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
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
function pmsymadd!(A::PeriodicMatrix, α = 1)
    m, n = size(A) 
    m == n || throw(ArgumentError("matrix A must be square"))
    if α == 1
       for i = 1:length(A) 
           inplace_transpose_add!(view(A.M[i],:,:))
       end
    else
        for i = 1:length(A) 
            inplace_transpose_add!(view(A.M[i],:,:),α)
        end
    end
    return A
end
function inplace_transpose_add!(A, α = 1)
    # compute (A+transpose(A))*α
    n = size(A, 1)
    if α == 1
        for i in 1:n
            for j in i:n
                A[i, j] += A[j, i]
                A[j, i] = A[i, j]
            end
        end
    else
        for i in 1:n
            for j in i:n
                A[i, j] += A[j, i]
                A[i,j]  *= α
                A[j, i] = A[i, j]
            end
        end
    end
end

function *(A::PeriodicMatrix, B::PeriodicMatrix)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    pa = length(A) 
    pb = length(B)
    nta = numerator(rationalize(period/A.period))
    ntb = numerator(rationalize(period/B.period))
    K = max(nta*A.nperiod*pa,ntb*B.nperiod*pb)
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
*(J::UniformScaling{<:Real}, A::PeriodicMatrix) = J.λ*A
*(A::PeriodicMatrix, J::UniformScaling{<:Real}) = A*J.λ
for (PMF, MF) in ((:pmmuladdsym, :muladdsym!), (:pmmultraddsym, :multraddsym!), (:pmmuladdtrsym,:muladdtrsym!) )
    @eval begin
        function $PMF(A::PeriodicMatrix,B::PeriodicMatrix,C::PeriodicMatrix, (α,β) = (true, true))
            if isconstant(A)
               isconstant(B) || isconstant(C) || B.Ts ≈ C.Ts || error("B and C must have the same sampling time")
            elseif isconstant(B)
               isconstant(C) || A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
            elseif isconstant(C)
                A.Ts ≈ B.Ts || error("A and B must have the same sampling time") 
            else
                A.Ts ≈ B.Ts ≈ C.Ts || error("A, B and C must have the same sampling time")
            end
            period = promote_period(A, B, C)
            pa = length(A) 
            pb = length(B)
            pc = length(C)
            p = lcm(pa,pb,pc)
            T = promote_type(eltype(A),eltype(B),eltype(B))
            X = Vector{Matrix{T}}(undef, p)
            nta = numerator(rationalize(period/A.period))
            K = nta*A.nperiod*pa
        
            for i = 1:p
                ia = mod(i-1,pa)+1
                ib = mod(i-1,pb)+1
                ic = mod(i-1,pc)+1
                # size(A.M[ia],2) == size(B.M[ib],1) || throw(DimensionMismatch("A and B have incompatible dimensions"))
                X[i] = copy(A.M[ia])
                $MF(view(X[i],:,:), B.M[ib], C.M[ic],(α,β))
            end
            return PeriodicMatrix(X, period; nperiod = div(K,p))
        end
        $PMF(A::PeriodicMatrix,B::PeriodicMatrix,C::PeriodicMatrix, α, β) = $PMF(A, B, C, (α,β))
    end
end
muladdsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, α, β) = muladdsym!(A,B,C,(α,β))
function muladdsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, (α,β) = (true, true))
    # compute in A the symmetrix matrix α*A +  β*B*C
    n = LinearAlgebra.checksquare(A)
    n == size(B,1) || throw(ArgumentError("A and B must have the same number of rows"))
    n == size(C,2) || throw(ArgumentError("A and C must have the same number of columns"))
    m = size(B,2)
    m == size(C,1) || throw(ArgumentError("B and C have incompatible dimensions"))
    ZERO = zero(promote_type(eltype(B),eltype(C)))
    if α == 0
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[i,k]*C[k,j])
                end
                A[i,j] = β*temp
                A[j,i] = A[i,j]
            end
        end
    else
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[i,k]*C[k,j])
                end
                A[i,j] = α*A[i,j]+β*temp
                A[j,i] = A[i,j]
            end
        end
    end
    return A
end
multraddsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, α, β) = multraddsym!(A, B, C, (α,β))
function multraddsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, (α,β) = (true, true))
    # compute in A the symmetrix matrix α*A +  β*transpose(B)*C
    n = LinearAlgebra.checksquare(A)
    n == size(B,2) || throw(ArgumentError("A and B must have the same number of columns"))
    n == size(C,2) || throw(ArgumentError("A and C must have the same number of columns"))
    m = size(B,1)
    m == size(C,1) || throw(ArgumentError("B and C have incompatible dimensions"))
    ZERO = zero(promote_type(eltype(B),eltype(C)))
    if α == 0
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[k,i]*C[k,j])
                end
                A[i,j] = β*temp
                A[j,i] = A[i,j]
            end
        end
    else
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[k,i]*C[k,j])
                end
                A[i,j] = α*A[i,j]+β*temp
                A[j,i] = A[i,j]
            end
        end
    end
    return A
end
muladdtrsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, α, β) = muladdtrsym!(A, B, C, (α,β))
function muladdtrsym!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, (α,β) = (true, true))
    # compute in A the symmetrix matrix α*A +  β*transpose(B)*C
    n = LinearAlgebra.checksquare(A)
    n == size(B,1) || throw(ArgumentError("A and B must have the same number of rows"))
    n == size(C,1) || throw(ArgumentError("A and C must have the same number of rows"))
    m = size(B,2)
    m == size(C,2) || throw(ArgumentError("B and C have incompatible dimensions"))
    ZERO = zero(promote_type(eltype(B),eltype(C)))
    if α == 0
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[i,k]*C[j,k])
                end
                A[i,j] = β*temp
                A[j,i] = A[i,j]
            end
        end
    else
        for i = 1:n
            for j = i:n
                temp = ZERO
                for k = 1:m
                    temp += (B[i,k]*C[j,k])
                end
                A[i,j] = α*A[i,j]+β*temp
                A[j,i] = A[i,j]
            end
        end
    end
    return A
end
for (PMF, MF) in ((:pmata, :mulatasym), (:pmaat, :mulaatsym) )
    @eval begin
        function $PMF(A::PeriodicMatrix)
            p = length(A) 
            T = eltype(A)
            X = Vector{Matrix{T}}(undef, p)
            for i = 1:p
                X[i] = $MF(view(A.M[i],:,:))
            end
            return PeriodicMatrix(X, A.period; nperiod = A.nperiod)
        end
    end
end
for (PMF, MF) in ((:pmata, :mulatasym), (:pmaat, :mulaatsym) )
    @eval begin
        function $PMF(A::PeriodicArray)
            m, n, p = size(A.M) 
            T = eltype(A)
            mn = $PMF == pmata ? n : m
            X = PeriodicArray(Array{T,3}(undef, mn, mn, p), A.period; nperiod = A.nperiod) 
            for i = 1:p
                copyto!(view(X.M,:,:,i), $MF(view(A.M,:,:,i)))
            end
            return X
        end
    end
end

        
function mulatasym(A::AbstractMatrix{T}) where {T}
    # compute the symmetrix matrix transpose(A)*A
    m, n = size(A,1), size(A,2) 
    X = similar(A,n,n)    
    ZERO = zero(T)
    for i = 1:n
        for j = i:n
            temp = ZERO
            for k = 1:m
                temp += (A[k,i]*A[k,j])
            end
            X[i,j] = temp
            X[j,i] = temp
        end
    end
    return X
end
function mulaatsym(A::AbstractMatrix{T}) where {T}
    # compute the symmetrix matrix A*transpose(A)
    m, n = size(A,1), size(A,2) 
    X = similar(A,m,m)    
    ZERO = zero(T)
    for i = 1:m
        for j = i:m
            temp = ZERO
            for k = 1:n
                temp += (A[i,k]*A[j,k])
            end
            X[i,j] = temp
            X[j,i] = temp
        end
    end
    return X
end


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

function horzcat(A::PeriodicMatrix, B::PeriodicMatrix)
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
        size(A.M[ia],1) == size(B.M[ib],1) || throw(DimensionMismatch("A and B have incompatible dimension"))
        X[i] = [A.M[ia] B.M[ib]]
    end
    return PeriodicMatrix(X, period; nperiod = div(K,p))
end
Base.hcat(A::PeriodicMatrix, B::PeriodicMatrix) = horzcat(A,B)
function vertcat(A::PeriodicMatrix, B::PeriodicMatrix)
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
        size(A.M[ia],2) == size(B.M[ib],2) || throw(DimensionMismatch("A and B have incompatible dimension"))
        X[i] = [A.M[ia]; B.M[ib]]
    end
    return PeriodicMatrix(X, period; nperiod = div(K,p))
end
Base.vcat(A::PeriodicMatrix, B::PeriodicMatrix) = vertcat(A,B)

function blockdiag(A::PeriodicMatrix, B::PeriodicMatrix)
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
        X[i] = DescriptorSystems.blockdiag(A.M[ia], B.M[ib])
    end
    return PeriodicMatrix(X, period; nperiod = div(K,p))
end


#  SwitchingPeriodicMatrix
function pmshift(A::SwitchingPeriodicMatrix, k::Int = 1)
    return convert(SwitchingPeriodicMatrix,pmshift(convert(PeriodicMatrix,A),k))
end
LinearAlgebra.inv(A::SwitchingPeriodicMatrix) = SwitchingPeriodicMatrix(inv.(A.M), A.ns, A.period; nperiod = A.nperiod)
function LinearAlgebra.transpose(A::SwitchingPeriodicMatrix)
    return SwitchingPeriodicMatrix(copy.(transpose.(A.M)), A.ns, A.period; nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::SwitchingPeriodicMatrix)
    return SwitchingPeriodicMatrix(copy.(transpose.(A.M)), A.ns, A.period; nperiod = A.nperiod)
end
function Base.reverse(A::SwitchingPeriodicMatrix)
    n = length(A.ns)
    return SwitchingPeriodicMatrix(reverse(A.M), n == 1 ? A.ns : [A.ns[n].-reverse(A.ns[1:n-1]); A.ns[n]], A.period; nperiod = A.nperiod)
end
function LinearAlgebra.opnorm(A::SwitchingPeriodicMatrix, p::Union{Real,Missing} = missing)
    if ismissing(p)
       return SwitchingPeriodicMatrix([norm(A.M[i]) for i in 1:length(A.M)], A.ns, A.period; nperiod = A.nperiod)
    else
       return SwitchingPeriodicMatrix([opnorm(A.M[i],p) for i in 1:length(A.M)], A.ns, A.period; nperiod = A.nperiod)
    end
end
function LinearAlgebra.norm(A::SwitchingPeriodicMatrix, p::Real = 2)
    k = length(A)
    k == 0 && (return zero(eltype(A)))
    tn = norm.(A.M)
    if p == 2
       tn[1] *= sqrt(A.ns[1]) 
       for i = 2:k
           tn[i] *= sqrt(A.ns[i]-A.ns[i-1])
       end
       return norm(tn,p)*sqrt(A.nperiod)
    elseif p == 1
       tn[1] *= A.ns[1] 
       for i = 2:k
           tn[i] *= (A.ns[i]-A.ns[i-1])
       end
       return norm(tn,p)*A.nperiod
    else 
       isinf(p) || throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
       return norm(tn,p) 
    end
end
function LinearAlgebra.tr(A::SwitchingPeriodicMatrix)
    return SwitchingPeriodicMatrix([[tr(A.M[i])] for i in 1:length(A.M)], A.ns, A.period; nperiod = A.nperiod)
end
function trace(A::SwitchingPeriodicMatrix) 
    t = zero(eltype(A))
    k = length(A)
    k == 0 && (return t)
    t += tr(A.M[1])*A.ns[1]
    for i in 2:k
        t += tr(A.M[i])*(A.ns[i]-A.ns[i-1])
    end
    return t*A.nperiod
end

Base.iszero(A::SwitchingPeriodicMatrix) = iszero(A.M)
LinearAlgebra.issymmetric(A::SwitchingPeriodicMatrix) = all([issymmetric(A.M[i]) for i in 1:length(A.ns)])
function ==(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    isconstant(A) && isconstant(B) && (return isequal(A.M, B.M))
    na = length(A.ns); nb = length(B.ns)
    if na == nb
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && isequal(A.M, B.M) 
    else
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && all([isequal(A[i],B[i]) for i in 1:max(na,nb)])
    end
end
function Base.isapprox(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.M, B.M; rtol, atol))
    na = length(A.ns); nb = length(B.ns)
    if na == nb
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && isapprox(A.M, B.M; rtol, atol) 
    else
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && all([isapprox(A[i],B[i]; rtol, atol) for i in 1:max(na,nb)])
    end
end
function Base.isapprox(A::SwitchingPeriodicMatrix, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(A.M[i], J; kwargs...) for i in 1:length(A.M)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::SwitchingPeriodicMatrix; kwargs...) = isapprox(A, J; kwargs...)

function +(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
    (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[i]+B.M[i] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    isconstant(A) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[1]+B.M[i] for i in 1:length(B.M)], B.ns, B.period, B.nperiod))
    isconstant(B) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[i]+B.M[1] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for addition")
    nperiod = A.nperiod
    if  nperiod == B.nperiod
        ns = unique(sort([A.ns;B.ns]))
    else
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                          vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
    end
    N = length(ns)                  
    return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([kpmeval(A,ns[i])+kpmeval(B,ns[i]) for i in 1:N], ns, A.period, nperiod)   
end
+(A::SwitchingPeriodicMatrix, C::AbstractMatrix) = +(A, SwitchingPeriodicMatrix([C], [A.dperiod], A.period))
+(A::AbstractMatrix, C::SwitchingPeriodicMatrix) = +(SwitchingPeriodicMatrix([A], [C.dperiod], C.period), C)
-(A::SwitchingPeriodicMatrix) = SwitchingPeriodicMatrix{:d,eltype(A)}(-A.M, A.ns, A.period, A.nperiod)
-(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix) = +(A,-B)
-(A::SwitchingPeriodicMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::SwitchingPeriodicMatrix) = +(A, -C)
function (+)(A::SwitchingPeriodicMatrix, J::UniformScaling{<:Real}) 
    nv, mv = size(A)
    n = minimum(nv) 
    n == maximum(nv) == minimum(mv) == maximum(mv) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::SwitchingPeriodicMatrix) = +(A,J)
(-)(A::SwitchingPeriodicMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::SwitchingPeriodicMatrix) = +(-A,J)

function *(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
    (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[i]*B.M[i] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    isconstant(A) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[1]*B.M[i] for i in 1:length(B.M)], B.ns, B.period, B.nperiod))
    isconstant(B) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([A.M[i]*B.M[1] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for multiplication")
    nperiod = A.nperiod
    if  nperiod == B.nperiod
        ns = unique(sort([A.ns;B.ns]))
    else
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                          vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
    end
    N = length(ns)                  
    return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([kpmeval(A,ns[i])*kpmeval(B,ns[i]) for i in 1:N], ns, A.period, nperiod)   
end
*(A::SwitchingPeriodicMatrix, C::AbstractMatrix) = SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(C))}([A.M[i]*C for i in 1:length(A.M)], A.ns, A.period, A.nperiod)
*(A::AbstractMatrix, C::SwitchingPeriodicMatrix) = SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(C))}([A*C.M[i] for i in 1:length(C.M)], C.ns, C.period, C.nperiod)
*(A::SwitchingPeriodicMatrix, C::Real) = SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(C))}([A.M[i]*C for i in 1:length(A.M)], A.ns, A.period, A.nperiod)
*(C::Real, A::SwitchingPeriodicMatrix) = SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(C))}([C*A.M[i] for i in 1:length(A.M)], A.ns, A.period, A.nperiod)
/(A::SwitchingPeriodicMatrix, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::SwitchingPeriodicMatrix) = J.λ*A
*(A::SwitchingPeriodicMatrix, J::UniformScaling{<:Real}) = A*J.λ

function horzcat(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
        (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[i] B.M[i]] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    isconstant(A) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[1] B.M[i]] for i in 1:length(B.M)], B.ns, B.period, B.nperiod))
    isconstant(B) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[i] B.M[1]] for i in 1:length(Av)], A.ns, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for horizontal concatenation")
    nperiod = A.nperiod
    if  nperiod == B.nperiod
        ns = unique(sort([A.ns;B.ns]))
    else
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                          vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
    end
    N = length(ns)                  
    return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[kpmeval(A,ns[i]) kpmeval(B,ns[i])] for i in 1:N], ns, A.period, nperiod)   
end
Base.hcat(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix) = horzcat(A,B)

function vertcat(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
        (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[i]; B.M[i]] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    isconstant(A) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[1]; B.M[i]] for i in 1:length(B.M)], B.ns, B.period, B.nperiod))
    isconstant(B) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[A.M[i]; B.M[1]] for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for vertical concatenation")
    nperiod = A.nperiod
    if  nperiod == B.nperiod
        ns = unique(sort([A.ns;B.ns]))
    else
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                          vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
    end
    N = length(ns)                  
    return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([[kpmeval(A,ns[i]); kpmeval(B,ns[i])] for i in 1:N], ns, A.period, nperiod)   
end
Base.vcat(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix) = vertcat(A,B)

function blockdiag(A::SwitchingPeriodicMatrix, B::SwitchingPeriodicMatrix)
    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
        (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.M[i], B.M[i]) for i in 1:length(A.M)], A.ns, A.period, A.nperiod))
    isconstant(A) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.M[1], B.M[i]) for i in 1:length(B.M)], B.ns, B.period, B.nperiod))
    isconstant(B) && 
       (return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.M[i], B.M[1]) for i in 1:length(A.M)], A.ns, A.period, A.nperiod))

    A.period == B.period || error("periods must be equal for block-diagonal appending")
    nperiod = A.nperiod
    if  nperiod == B.nperiod
        ns = unique(sort([A.ns;B.ns]))
    else
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                          vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
    end
    N = length(ns)                  
    return SwitchingPeriodicMatrix{:d,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(kpmeval(A,ns[i]), kpmeval(B,ns[i])) for i in 1:N], ns, A.period, nperiod)   
end

#  SwitchingPeriodicArray
function pmshift(A::SwitchingPeriodicArray, k::Int = 1)
    return convert(SwitchingPeriodicArray,pmshift(convert(PeriodicArray,A),k))
end
function LinearAlgebra.inv(A::SwitchingPeriodicArray)
    x = similar(A.M)
    [x[:,:,i] = inv(A.M[:,:,i]) for i in 1:size(A.M,3)]
    SwitchingPeriodicArray(x, A.ns, A.period; nperiod = A.nperiod)
end
function LinearAlgebra.transpose(A::SwitchingPeriodicArray)
    return SwitchingPeriodicArray(permutedims(A.M,(2,1,3)), A.ns, A.period; nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::SwitchingPeriodicArray)
    return SwitchingPeriodicArray(permutedims(A.M,(2,1,3)), A.ns, A.period; nperiod = A.nperiod)
end
function Base.reverse(A::SwitchingPeriodicArray)
    n = length(A.ns)
    return SwitchingPeriodicArray(reverse(A.M,dims=3), n == 1 ? A.ns : [A.ns[n].-reverse(A.ns[1:n-1]); A.ns[n]], A.period; nperiod = A.nperiod)
end
function LinearAlgebra.opnorm(A::SwitchingPeriodicArray, p::Union{Real,Missing} = missing)
    k = size(A.M,3)
    x = Array{eltype(A),3}(undef, 1, 1, k)
    if ismissing(p)
        [x[1,1,i] = norm(view(A.M,:,:,i)) for i in 1:k]  # Frobenius noorm
    else
        [x[1,1,i] = opnorm(view(A.M,:,:,i),p) for i in 1:k] # p-norm
    end
    return SwitchingPeriodicArray(x, A.ns, A.period; nperiod = A.nperiod)
end
function LinearAlgebra.norm(A::SwitchingPeriodicArray, p::Real = 2)
    k = length(A)
    k == 0 && (return zero(eltype(A)))
    tn = [norm(view(A.M,:,:,i)) for i in 1:k]
    if p == 2
       tn[1] *= sqrt(A.ns[1]) 
       for i = 2:k
           tn[i] *= sqrt(A.ns[i]-A.ns[i-1])
       end
       return norm(tn,p)*sqrt(A.nperiod)
    elseif p == 1
       tn[1] *= A.ns[1] 
       for i = 2:k
           tn[i] *= (A.ns[i]-A.ns[i-1])
       end
       return norm(tn,p)*A.nperiod
    else 
       isinf(p) || throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
       return norm(tn,p) 
    end
end
function LinearAlgebra.tr(A::SwitchingPeriodicArray)
    p = size(A.M,3)
    x = Array{eltype(A),3}(undef, 1, 1, p)
    [x[1,1,i] = tr(view(A.M,:,:,i)) for i in 1:p]
    return SwitchingPeriodicArray(x, A.ns, A.period; nperiod = A.nperiod)
end
function trace(A::SwitchingPeriodicArray) 
    t = zero(eltype(A))
    k = length(A)
    k == 0 && (return t)
    t += tr(view(A.M,:,:,1))*A.ns[1]
    for i in 2:k
        t += tr(view(A.M,:,:,i))*(A.ns[i]-A.ns[i-1])
    end
    return t*A.nperiod
end

LinearAlgebra.issymmetric(A::SwitchingPeriodicArray) = all([issymmetric(view(A.M,:,:,i)) for i in 1:length(A.ns)])
Base.iszero(A::SwitchingPeriodicArray) = iszero(A.M)
function ==(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) && isconstant(B) && (return isequal(A.M, B.M))
    na = length(A.ns); nb = length(B.ns)
    if na == nb
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && isequal(A.M, B.M) 
    else
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && all([isequal(A[i],B[i]) for i in 1:max(na,nb)])
    end
end
function Base.isapprox(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.M, B.M; rtol, atol))
    na = length(A.ns); nb = length(B.ns)
    if na == nb
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && isapprox(A.M, B.M; rtol, atol) 
    else
       (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && all([isapprox(A[i],B[i]; rtol, atol) for i in 1:max(na,nb)])
    end
end
function Base.isapprox(A::SwitchingPeriodicArray, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(view(A.M,:,:,i), J; kwargs...) for i in 1:length(A.ns)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::SwitchingPeriodicArray; kwargs...) = isapprox(A, J; kwargs...)

function +(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    T = promote_type(eltype(A),eltype(B))
    m, n, pa = size(A.M)
    mb, nb, pb = size(B.M)
    (m, n) == (mb, nb) || throw(DimensionMismatch("A and B must have the same dimensions"))

    A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns &&
    (return SwitchingPeriodicArray{:d,T}(A.M+B.M, A.ns, A.period, A.nperiod))
    if isconstant(A) 
        N = length(B.ns)
        X = Array{T,3}(undef, m, n, N)
        for i = 1:N
            X[:,:,i] = A.M[:,:,1]+B.M[:,:,i]
        end
        return SwitchingPeriodicArray{:d,T}(X, B.ns, B.period, B.nperiod)
    elseif isconstant(B) 
        N = length(A.ns)
        X = Array{T,3}(undef, m, n, N)
        for i = 1:N
            X[:,:,i] = A.M[:,:,i]+B.M[:,:,1]
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    else
        A.period == B.period || error("periods must be equal for addition")
        nperiod = A.nperiod
        if nperiod == B.nperiod
           ns = unique(sort([A.ns;B.ns]))
        else
           nperiod = gcd(A.nperiod,B.nperiod)
           ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                             vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
        end
        N = length(ns)   
        X = Array{T,3}(undef, m, n, N)
        for i = 1:N
            X[:,:,i] = kpmeval(A,ns[i])+kpmeval(B,ns[i])
        end
        return SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(B))}(X, ns, A.period, nperiod)   
    end              
end
+(A::SwitchingPeriodicArray, C::AbstractMatrix) = +(A, SwitchingPeriodicArray(reshape(C,size(A,1),size(A,2),1), [A.dperiod], A.period))
+(A::AbstractMatrix, C::SwitchingPeriodicArray) = +(SwitchingPeriodicArray([A], C.ns, [C.dperiod], C.period), C)
-(A::SwitchingPeriodicArray) = SwitchingPeriodicArray{:d,eltype(A)}(-A.M, A.ns, A.period, A.nperiod)
-(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray) = +(A,-B)
-(A::SwitchingPeriodicArray, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::SwitchingPeriodicArray) = +(A, -C)
function (+)(A::SwitchingPeriodicArray, J::UniformScaling{<:Real}) 
    m, n = size(A)
    n == m || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::SwitchingPeriodicArray) = +(A,J)
(-)(A::SwitchingPeriodicArray, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::SwitchingPeriodicArray) = +(-A,J)

function *(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    T = promote_type(eltype(A),eltype(B))
    m, n, pa = size(A.M)
    mb, nb, pb = size(B.M)
    n == mb || throw(DimensionMismatch("number of columns of A $n not equal to the number of rows of B $mb"))

    if A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns 
        N = length(A.ns)
        X = Array{T,3}(undef, m, nb, N)
        for i = 1:N
            mul!(view(X,:,:,i),view(A.M,:,:,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    end
    if isconstant(A) 
        N = length(B.ns)
        X = Array{T,3}(undef, m, nb, N)
        for i = 1:N
            mul!(view(X,:,:,i),view(A.M,:,:,1),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, B.ns, B.period, B.nperiod)
    elseif isconstant(B) 
        N = length(A.ns)
        X = Array{T,3}(undef, m, n, N)
        for i = 1:N
            mul!(view(X,:,:,i),view(A.M,:,:,i),view(B.M,:,:,1))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    else
        A.period == B.period || error("periods must be equal for addition")
        nperiod = A.nperiod
        if nperiod == B.nperiod
           ns = unique(sort([A.ns;B.ns]))
        else
           nperiod = gcd(A.nperiod,B.nperiod)
           ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                             vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
        end
        N = length(ns)   
        X = Array{T,3}(undef, m, nb, N)
        for i = 1:N
            mul!(view(X,:,:,i),kpmeval(A,ns[i]),kpmeval(B,ns[i]))
        end
        return SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(B))}(X, ns, A.period, nperiod)   
    end
end
*(A::SwitchingPeriodicArray, C::AbstractMatrix) = *(A, SwitchingPeriodicArray(reshape(C,size(A,1),size(A,2),1), [A.dperiod], A.period))
*(A::AbstractMatrix, C::SwitchingPeriodicArray) = *(SwitchingPeriodicArray([A], C.ns, [C.dperiod], C.period), C)
# *(A::SwitchingPeriodicArray, C::AbstractMatrix) = SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(C))}([A.M[i]*C for i in 1:length(A.M)], A.ns, A.period, A.nperiod)
# *(A::AbstractMatrix, C::SwitchingPeriodicArray) = SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(C))}([A*C.M[i] for i in 1:length(C.M)], C.ns, C.period, C.nperiod)
*(A::SwitchingPeriodicArray, C::Real) = SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(C))}(A.M*C, A.ns, A.period, A.nperiod)
*(C::Real, A::SwitchingPeriodicArray) = SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(C))}(C*A.M, A.ns, A.period, A.nperiod)
/(A::SwitchingPeriodicArray, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::SwitchingPeriodicArray) = J.λ*A
*(A::SwitchingPeriodicArray, J::UniformScaling{<:Real}) = A*J.λ


function horzcat(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    T = promote_type(eltype(A),eltype(B))
    ma, na, pa = size(A.M)
    mb, nb, pb = size(B.M)
    ma == mb || throw(DimensionMismatch("number of rows of A $ma not equal to the number of rows of B $mb"))

    if A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns 
        N = length(A.ns)
        X = Array{T,3}(undef, ma, na+nb, N)
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,:,j1,i),view(A.M,:,:,i))
            copyto!(view(X,:,j2,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    end
    if isconstant(A) 
        N = length(B.ns)
        X = Array{T,3}(undef, ma, na+nb, N)
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,:,j1,i),view(A.M,:,:,1))
            copyto!(view(X,:,j2,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, B.ns, B.period, B.nperiod)
    elseif isconstant(B) 
        N = length(A.ns)
        X = Array{T,3}(undef, ma, na+nb, N)
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,:,j1,i),view(A.M,:,:,i))
            copyto!(view(X,:,j2,i),view(B.M,:,:,1))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    else
        A.period == B.period || error("periods must be equal for addition")
        nperiod = A.nperiod
        if nperiod == B.nperiod
           ns = unique(sort([A.ns;B.ns]))
        else
           nperiod = gcd(A.nperiod,B.nperiod)
           ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                             vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
        end
        N = length(ns)   
        X = Array{T,3}(undef, ma, na+nb, N)
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,:,j1,i),kpmeval(A,ns[i]))
            copyto!(view(X,:,j2,i),kpmeval(B,ns[i]))
        end
        return SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(B))}(X, ns, A.period, nperiod)   
    end
end
Base.hcat(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray) = horzcat(A,B)

function vertcat(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    T = promote_type(eltype(A),eltype(B))
    ma, na, pa = size(A.M)
    mb, nb, pb = size(B.M)
    na == nb || throw(DimensionMismatch("number of columnss of A $na not equal to the number of columns of B $nb"))

    if A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns 
        N = length(A.ns)
        X = Array{T,3}(undef, ma+mb, na, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        for i = 1:N
            copyto!(view(X,i1,:,i),view(A.M,:,:,i))
            copyto!(view(X,i2,:,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    end
    if isconstant(A) 
        N = length(B.ns)
        X = Array{T,3}(undef, ma+mb, na, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        for i = 1:N
            copyto!(view(X,i1,:,i),view(A.M,:,:,1))
            copyto!(view(X,i2,:,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, B.ns, B.period, B.nperiod)
    elseif isconstant(B) 
        N = length(A.ns)
        X = Array{T,3}(undef, ma+mb, na, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        for i = 1:N
            copyto!(view(X,i1,:,i),view(A.M,:,:,i))
            copyto!(view(X,i2,:,i),view(B.M,:,:,1))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    else
        A.period == B.period || error("periods must be equal for addition")
        nperiod = A.nperiod
        if nperiod == B.nperiod
           ns = unique(sort([A.ns;B.ns]))
        else
           nperiod = gcd(A.nperiod,B.nperiod)
           ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                             vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
        end
        N = length(ns)   
        X = Array{T,3}(undef, ma+mb, na, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        for i = 1:N
            copyto!(view(X,i1,:,i),kpmeval(A,ns[i]))
            copyto!(view(X,i2,:,i),kpmeval(B,ns[i]))
        end
        return SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(B))}(X, ns, A.period, nperiod)   
    end
end
Base.vcat(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray) = vertcat(A,B)

function blockdiag(A::SwitchingPeriodicArray, B::SwitchingPeriodicArray)
    isconstant(A) || isconstant(B) || A.Ts ≈ B.Ts || error("A and B must have the same sampling time")
    period = promote_period(A, B)
    T = promote_type(eltype(A),eltype(B))
    ma, na, pa = size(A.M)
    mb, nb, pb = size(B.M)

    if A.period == B.period && A.nperiod == B.nperiod && A.ns == B.ns 
        N = length(A.ns)
        X = zeros(T, ma+mb, na+nb, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,i1,j1,i),view(A.M,:,:,i))
            copyto!(view(X,i2,j2,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    end
    if isconstant(A) 
        N = length(B.ns)
        X = zeros(T, ma+mb, na+nb, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,i1,j1,i),view(A.M,:,:,1))
            copyto!(view(X,i2,j2,i),view(B.M,:,:,i))
        end
        return SwitchingPeriodicArray{:d,T}(X, B.ns, B.period, B.nperiod)
    elseif isconstant(B) 
        N = length(A.ns)
        X = zeros(T, ma+mb, na+nb, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,i1,j1,i),view(A.M,:,:,i))
            copyto!(view(X,i2,j2,i),view(B.M,:,:,1))
        end
        return SwitchingPeriodicArray{:d,T}(X, A.ns, A.period, A.nperiod)
    else
        A.period == B.period || error("periods must be equal for addition")
        nperiod = A.nperiod
        if nperiod == B.nperiod
           ns = unique(sort([A.ns;B.ns]))
        else
           nperiod = gcd(A.nperiod,B.nperiod)
           ns = unique(sort([vcat([(i-1)*A.dperiod .+ A.ns for i in 1:div(A.nperiod,nperiod)]...);
                             vcat([(i-1)*B.dperiod .+ B.ns for i in 1:div(B.nperiod,nperiod)]...)]))
        end
        N = length(ns)   
        X = zeros(T, ma+mb, na+nb, N)
        i1 = 1:ma; i2 = ma+1:ma+mb
        j1 = 1:na; j2 = na+1:na+nb
        for i = 1:N
            copyto!(view(X,i1,j1,i),kpmeval(A,ns[i]))
            copyto!(view(X,i2,j2,i),kpmeval(B,ns[i]))
        end
        return SwitchingPeriodicArray{:d,promote_type(eltype(A),eltype(B))}(X, ns, A.period, nperiod)   
    end
end


# Operations with periodic function matrices
function derivative(A::PeriodicFunctionMatrix{:c,T};  h::Union{Missing,Real} = missing,  method = "cd") where {T}
    isconstant(A) && (return PeriodicFunctionMatrix{:c,T}(t -> zeros(T,A.dims...), A.period, A.dims, A.nperiod, true))
    # centered differences
    if method == "cd" 
       ismissing(h) && (h = A.period*sqrt(eps(T)))
       return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t-h))/(2*h), A.period, A.dims, A.nperiod, false)
    end
    # fourth-order differences
    if method == "4d" 
        ismissing(h) && (h = A.period*sqrt(sqrt(eps(T))))
        return PeriodicFunctionMatrix{:c,T}(t -> (-A.f(t+2h) + A.f(t-2h)  + 8*(A.f(t+h) - A.f(t-h)))/(12h), A.period, A.dims, A.nperiod, false)
    end
    # first-order differences 
    ismissing(h) && (h = A.period*sqrt(eps(T)))
    return PeriodicFunctionMatrix{:c,T}(t -> (A.f(t+h)-A.f(t))/h, A.period, A.dims, A.nperiod, false)
end
function LinearAlgebra.inv(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> inv(A.f(t)), A.period, A.dims, A.nperiod, A._isconstant)
end
function LinearAlgebra.tr(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> [tr(A.f(t))], A.period, (1,1), A.nperiod, A._isconstant)
end
function trace(A::PeriodicFunctionMatrix; K = 128) 
    isconstant(A) && (return tr(A.f(0)))
    t = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    tt = zero(eltype(Δ))
    for i = 1:K
        tt += tr(A.f(t))*Δ
        t += Δ
    end 
    return tt*A.nperiod/A.period
end
function LinearAlgebra.transpose(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> transpose(A.f(t)), A.period, (A.dims[2],A.dims[1]), A.nperiod, A._isconstant)
end
function LinearAlgebra.adjoint(A::PeriodicFunctionMatrix{:c,T})  where {T}
    return PeriodicFunctionMatrix{:c,T}(t -> adjoint(A.f(t)), A.period, (A.dims[2],A.dims[1]), A.nperiod, A._isconstant)
end
function LinearAlgebra.opnorm(A::PeriodicFunctionMatrix, p::Union{Real, Missing} = missing) 
    if ismissing(p)
       return PeriodicFunctionMatrix{:c,eltype(A)}(t -> [norm(A.f(t))], A.period, (1,1), A.nperiod, A._isconstant)
    else
       return PeriodicFunctionMatrix{:c,eltype(A)}(t -> [opnorm(A.f(t),p)], A.period, (1,1), A.nperiod, A._isconstant)
    end
end
function norm(A::PeriodicFunctionMatrix, p::Real = 2; K = 128) 
    isconstant(A) && (return norm(A.f(0)))
    nrm = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    ts = zero(eltype(Δ))
    if p == 2
       for i = 1:K
           nrm += norm(A.f(ts))^2*Δ
           ts += Δ
       end 
       return sqrt(nrm*A.nperiod)
    elseif isinf(p)
        for i = 1:K
            nrm = max(nrm,norm(A.f(ts)))
            ts += Δ
        end 
        return nrm
    elseif p == 1    
        for i = 1:K
            nrm += norm(A.f(ts))*Δ
            ts += Δ
        end 
        return nrm*A.nperiod
    else
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
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
    A.dims[2] == B.dims[1] || throw(DimensionMismatch("A and B have incompatible dimensions"))
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
*(J::UniformScaling{<:Real}, A::PeriodicFunctionMatrix) = J.λ*A
*(A::PeriodicFunctionMatrix, J::UniformScaling{<:Real}) = A*J.λ
for (PMF, MF) in ((:pmmuladdsym, :muladdsym!), (:pmmultraddsym, :multraddsym!), (:pmmuladdtrsym,:muladdtrsym!) )
    @eval begin
        function  $PMF(A::PeriodicFunctionMatrix,B::PeriodicFunctionMatrix,C::PeriodicFunctionMatrix, (α,β) = (true, true))
            period = promote_period(A, B, C)
            nperiod = gcd(A.nperiod,B.nperiod,C.nperiod)
            T = promote_type(eltype(A),eltype(B),eltype(B))
            if isconstant(A) && isconstant(B) && isconstant(C)
                return PeriodicFunctionMatrix{:c,T}(t -> $MF(A.f(0),B.f(0),C.f(0),(α,β)), period, (A.dims[1],A.dims[2]), nperiod, true)
             else
                return PeriodicFunctionMatrix{:c,T}(t -> $MF(A.f(t),B.f(t),C.f(t),(α,β)), period, (A.dims[1],B.dims[2]), nperiod, false)
             end
        end
    end
end
for PM in (:PeriodicFunctionMatrix, :PeriodicSymbolicMatrix, :HarmonicArray, :FourierFunctionMatrix, :PeriodicTimeSeriesMatrix)
    @eval begin
        pmmuladdsym(A::$PM,B::AbstractMatrix,C::$PM, (α,β) = (true, true)) = pmmuladdsym(A, $PM(B, A.Ts; nperiod = 1), C, (α,β))
        pmmuladdsym(A::$PM,B::$PM,C::AbstractMatrix, (α,β) = (true, true)) = pmmuladdsym(A, B, $PM(C, A.Ts; nperiod = 1), (α,β))
        pmmuladdsym(A::$PM,B::AbstractMatrix,C::AbstractMatrix, (α,β) = (true, true)) = pmmuladdsym(A, $PM(B, A.Ts; nperiod = 1), $PM(C, A.Ts; nperiod = 1), (α,β))
        pmmuladdsym(A::AbstractMatrix,B::$PM,C::$PM, (α,β) = (true, true)) = pmmuladdsym($PM(A, B.Ts; nperiod = 1), B, C, (α,β))
        pmmuladdsym(A::AbstractMatrix,B::AbstractMatrix,C::$PM, (α,β) = (true, true)) = pmmuladdsym($PM(A, C.Ts; nperiod = 1), $PM(B, C.Ts; nperiod = 1), C, (α,β))
        pmmuladdsym(A::AbstractMatrix,B::$PM,C::AbstractMatrix, (α,β) = (true, true)) = pmmuladdsym($PM(A, B.Ts; nperiod = 1), B, $PM(C, B.Ts; nperiod = 1), (α,β))
    end
end

#pmmuladdsym(A::AbstractMatrix,B::PeriodicFunctionMatrix,C::PeriodicFunctionMatrix, (α,β) = (true, true)) = pmmuladdsym(PeriodicFunctionMatrix(A, B.period), B, C, (α,β))
# function pmmultraddsym(A::PeriodicFunctionMatrix,B::PeriodicFunctionMatrix,C::PeriodicFunctionMatrix, (α,β) = (true, true))
#     period = promote_period(A, B, C)
#     nperiod = gcd(A.nperiod,B.nperiod,C.nperiod)
#     T = promote_type(eltype(A),eltype(B),eltype(B))
#     if isconstant(A) && isconstant(B) && isconstant(C)
#         return PeriodicFunctionMatrix{:c,T}(t ->  multraddsym!(A.f(0),B.f(0),C.f(0),(α,β)), period, (A.dims[1],A.dims[2]), nperiod, true)
#      else
#         return PeriodicFunctionMatrix{:c,T}(t -> multraddsym!(A.f(t),B.f(t),C.f(t),(α,β)), period, (A.dims[1],B.dims[2]), nperiod, false)
#      end
# end
#pmmultraddsym(A::AbstractMatrix,B::PeriodicFunctionMatrix,C::PeriodicFunctionMatrix, (α,β) = (true, true)) = pmmultraddsym(PeriodicFunctionMatrix(A, B.period), B, C, (α,β))


function horzcat(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    period = promote_period(A, B)
    A.dims[1] == B.dims[1] || throw(DimensionMismatch("A and B have incompatible row dimensions"))
    nperiod = gcd(A.nperiod,B.nperiod)
    T = promote_type(eltype(A),eltype(B))
    if isconstant(A) && isconstant(B)
       return PeriodicFunctionMatrix{:c,T}(t -> [A.f(0) B.f(0)], period, (A.dims[1],A.dims[2]+B.dims[2]), nperiod, true)
    else
       return PeriodicFunctionMatrix{:c,T}(t -> [A.f(t) B.f(t)], period, (A.dims[1],A.dims[2]+B.dims[2]), nperiod, false)
    end
end
horzcat(A::PeriodicFunctionMatrix, C::AbstractMatrix) = horzcat(A, PeriodicFunctionMatrix(C, A.period))
horzcat(A::AbstractMatrix, C::PeriodicFunctionMatrix) = horzcat(PeriodicFunctionMatrix(A, C.period), C)
Base.hcat(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix) = horzcat(A,B)
Base.hcat(A::PeriodicFunctionMatrix, B::AbstractMatrix) = horzcat(A,B)
Base.hcat(A::AbstractMatrix, B::PeriodicFunctionMatrix) = horzcat(A,B)

function vertcat(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    period = promote_period(A, B)
    A.dims[2] == B.dims[2] || throw(DimensionMismatch("A and B have incompatible column dimensions"))
    nperiod = gcd(A.nperiod,B.nperiod)
    T = promote_type(eltype(A),eltype(B))
    if isconstant(A) && isconstant(B)
       return PeriodicFunctionMatrix{:c,T}(t -> [A.f(0); B.f(0)], period, (A.dims[1]+B.dims[1],A.dims[2]), nperiod, true)
    else
       return PeriodicFunctionMatrix{:c,T}(t -> [A.f(t); B.f(t)], period, (A.dims[1]+B.dims[1],A.dims[2]), nperiod, false)
    end
end
Base.vcat(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix) = vertcat(A,B)

function blockdiag(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    T = promote_type(eltype(A),eltype(B))
    if isconstant(A) && isconstant(B)
       return PeriodicFunctionMatrix{:c,T}(t -> DescriptorSystems.blockdiag(A.f(0), B.f(0)), period, (A.dims[1]+B.dims[1],A.dims[2]+B.dims[2]), nperiod, true)
    else
       return PeriodicFunctionMatrix{:c,T}(t -> DescriptorSystems.blockdiag(A.f(t),B.f(t)), period, (A.dims[1]+B.dims[1],A.dims[2]+B.dims[2]), nperiod, false)
    end
end




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
#LinearAlgebra.inv(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(inv(A.F), A.period; nperiod = A.nperiod)
function LinearAlgebra.inv(A::PeriodicSymbolicMatrix)
    if isconstant(A)
       # fix for Symbolics.jl issue #895
       @variables t 
       PeriodicSymbolicMatrix(Num.(inv(Symbolics.unwrap.(substitute.(A.F, (Dict(t => 0.),))))), A.period; nperiod = A.nperiod)    
    else
       PeriodicSymbolicMatrix(inv(A.F), A.period; nperiod = A.nperiod)
    end
end
function LinearAlgebra.transpose(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(transpose(A.F)), A.period, nperiod = A.nperiod)
end
function LinearAlgebra.adjoint(A::PeriodicSymbolicMatrix)  
    return PeriodicSymbolicMatrix{:c,Num}(copy(adjoint(A.F)), A.period, nperiod = A.nperiod)
end
function Symbolics.simplify(A::PeriodicSymbolicMatrix)
    return PeriodicSymbolicMatrix{:c,Num}(Symbolics.simplify.(A.F), A.period; nperiod = A.nperiod)
end
LinearAlgebra.tr(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix([tr(A.F)], A.period; nperiod = A.nperiod)
function trace(A::PeriodicSymbolicMatrix; K = 128) 
    @variables t 
    trs = tr(A.F)  
    isconstant(A) && (return Symbolics.unwrap.(substitute.(trs, (Dict(t => 0),)))[1]*Δ)
    ts = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    tt = zero(eltype(Δ))
    for i = 1:K
        tt += Symbolics.unwrap.(substitute.(trs, (Dict(t => ts),)))[1]*Δ
        ts += Δ
    end 
    return tt*A.nperiod/A.period
end

function norm(A::PeriodicSymbolicMatrix, p::Real = 2; K = 128) 
    @variables t   
    isconstant(A) && (return norm(Symbolics.unwrap.(substitute.(A.F, (Dict(t => 0),)))))
    nrm = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    ts = zero(eltype(Δ))
    if p == 2
       for i = 1:K
           nrm += norm(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))))^2*Δ
           ts += Δ
       end 
       return sqrt(nrm*A.nperiod)
    elseif isinf(p)
        for i = 1:K
            nrm = max(nrm,norm(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),)))))
            ts += Δ
        end 
        return nrm
    elseif p == 1    
        for i = 1:K
            nrm += norm(Symbolics.unwrap.(substitute.(A.F, (Dict(t => ts),))))*Δ
            ts += Δ
        end 
        return nrm*A.nperiod
    else
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end 
function +(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    #return PeriodicSymbolicMatrix{:c,Num}(Symbolics.simplify.(A.F + B.F), period; nperiod)
    return PeriodicSymbolicMatrix{:c,Num}(A.F + B.F, period; nperiod)
end
+(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = +(A, PeriodicSymbolicMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = +(PeriodicSymbolicMatrix(A, C.period), C)
+(A::PeriodicSymbolicMatrix, C::PeriodicFunctionMatrix) = +(convert(PeriodicFunctionMatrix,A), C)
+(A::PeriodicFunctionMatrix, C::PeriodicSymbolicMatrix) = +(A, convert(PeriodicFunctionMatrix,C))
-(A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(-A.F, A.period; nperiod = A.nperiod)
-(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix) = +(A,-B)
-(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = +(A, -C)
-(A::PeriodicSymbolicMatrix, C::PeriodicFunctionMatrix) = +(A,-C)
-(A::PeriodicFunctionMatrix, C::PeriodicSymbolicMatrix) = +(A, -C)

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
*(A::PeriodicSymbolicMatrix, C::PeriodicFunctionMatrix) = *(convert(PeriodicFunctionMatrix,A), C)
*(A::PeriodicFunctionMatrix, C::PeriodicSymbolicMatrix) = *(A, convert(PeriodicFunctionMatrix,C))
*(A::PeriodicSymbolicMatrix, C::Real) = PeriodicSymbolicMatrix(C*A.F, A.period; nperiod = A.nperiod)
*(C::Real, A::PeriodicSymbolicMatrix) = PeriodicSymbolicMatrix(C*A.F, A.period; nperiod = A.nperiod)
/(A::PeriodicSymbolicMatrix, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::PeriodicSymbolicMatrix) = J.λ*A 
*(A::PeriodicSymbolicMatrix, J::UniformScaling{<:Real}) = A*J.λ

function horzcat(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    return PeriodicSymbolicMatrix{:c,Num}([A.F B.F], period; nperiod)
end
hcat(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix) = horzcat(A,B)
hcat(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = horzcat(A, PeriodicSymbolicMatrix(C, A.period))
hcat(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = horzcat(PeriodicSymbolicMatrix(A, C.period), C)
horzcat(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = horzcat(A, PeriodicSymbolicMatrix(C, A.period))
horzcat(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = horzcat(PeriodicSymbolicMatrix(A, C.period), C)


function vertcat(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    return PeriodicSymbolicMatrix{:c,Num}([A.F; B.F], period; nperiod)
end
vcat(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix) = vertcat(A,B)
vcat(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = vertcat(A, PeriodicSymbolicMatrix(C, A.period))
vcat(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = vertcat(PeriodicSymbolicMatrix(A, C.period), C)
vertcat(A::PeriodicSymbolicMatrix, C::AbstractMatrix) = vertcat(A, PeriodicSymbolicMatrix(C, A.period))
vertcat(A::AbstractMatrix, C::PeriodicSymbolicMatrix) = vertcat(PeriodicSymbolicMatrix(A, C.period), C)


function blockdiag(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix)
    period = promote_period(A, B)
    nperiod = gcd(A.nperiod,B.nperiod)
    return PeriodicSymbolicMatrix{:c,Num}(DescriptorSystems.blockdiag(A.F, B.F), period; nperiod)
end



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
function tr(A::HarmonicArray{:c,T}) where {T}
    l = size(A.values,3)
    Ahr = zeros(Complex{T}, 1, 1, l)
    for i = 1:l
        #Ahr[:,:,i] .= complex(tr(real(A.values[:,:,i])),tr(imag(A.values[:,:,i]))) 
        Ahr[1,1,i] = tr(A.values[:,:,i]) 
    end
    return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod)
end
function trace(A::HarmonicArray; K = 128) 
    isconstant(A) && (return tr(tpmeval(A, 0)))
    ts = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    tt = zero(eltype(Δ))
    for i = 1:K
        tt += tr(tpmeval(A, ts))*Δ
        ts += Δ
    end 
    return tt*A.nperiod/A.period
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
function LinearAlgebra.opnorm(A::HarmonicArray, p::Union{Real, Missing} = missing) 
    return convert(HarmonicArray,opnorm(convert(PeriodicFunctionMatrix,A),p))
end
function norm(A::HarmonicArray, p::Real = 2; K = 128) 
    isconstant(A) && (return norm(tpmeval(A, 0)))
    nrm = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    ts = zero(eltype(Δ))
    if p == 2
       for i = 1:K
           nrm += norm(tpmeval(A, ts))^2*Δ
           ts += Δ
       end 
       return sqrt(nrm*A.nperiod)
    elseif isinf(p)
        for i = 1:K
            nrm = max(nrm,norm(tpmeval(A, ts)))
            ts += Δ
        end 
        return nrm
    elseif p == 1    
        for i = 1:K
            nrm += norm(tpmeval(A, ts))*Δ
            ts += Δ
        end 
        return nrm*A.nperiod
    else
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end
function +(A::HarmonicArray, B::HarmonicArray)
    if A.period == B.period && A.nperiod == B.nperiod
       m, n, la = size(A.values)
       mb, nb, lb = size(B.values)
       (m, n) == (mb, nb) || throw(DimensionMismatch("A and B must have the same size"))
       T = promote_type(eltype(A.values),eltype(B.values))
       lmax = max(la,lb)
       Ahr = zeros(T,m,n,lmax)
       if la >= lb
          copyto!(view(Ahr,1:m,1:n,1:la),A.values) 
          #Ahr[:,:,1:la] = copy(A.values) 
          Ahr[:,:,1:lb] .+= view(B.values,1:m,1:n,1:lb)  
       else
          copyto!(view(Ahr,1:m,1:n,1:lb),B.values) 
          #Ahr = copy(B.values) 
          Ahr[:,:,1:la] .+= view(A.values,1:m,1:n,1:la)  
       end
       tol = 10*eps(real(T))*max(norm(A.values,Inf),norm(B.values,Inf)) 
       l = lmax
       for i = lmax:-1:2
           norm(Ahr[:,:,i],Inf) > tol && break
           l -= 1
       end
       l < lmax && (Ahr = Ahr[:,:,1:l])
       return HarmonicArray{:c,real(T)}(Ahr, A.period, nperiod = A.nperiod) 
    else
       #TODO: fix different numbers of subperiods 
       convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) + convert(PeriodicFunctionMatrix,B))
    end
end
+(A::HarmonicArray, C::AbstractMatrix) = +(A, HarmonicArray(C, A.period))
+(A::AbstractMatrix, C::HarmonicArray) = +(HarmonicArray(A, C.period), C)
+(A::HarmonicArray, C::PeriodicFunctionMatrix) = +(convert(PeriodicFunctionMatrix,A), C)
+(A::PeriodicFunctionMatrix, C::HarmonicArray) = +(A, convert(PeriodicFunctionMatrix,C))
-(A::HarmonicArray) = HarmonicArray(-A.values, A.period; nperiod = A.nperiod)
-(A::HarmonicArray, B::HarmonicArray) = +(A,-B)
-(A::HarmonicArray, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::HarmonicArray) = +(A, -C)
-(A::HarmonicArray, C::PeriodicFunctionMatrix) = +(A,-C)
-(A::PeriodicFunctionMatrix, C::HarmonicArray) = +(A, -C)
function (+)(A::HarmonicArray, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(A,J)
(-)(A::HarmonicArray, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::HarmonicArray) = +(-A,J)

*(A::HarmonicArray, B::HarmonicArray) = convert(HarmonicArray,convert(PeriodicFunctionMatrix,A) * convert(PeriodicFunctionMatrix,B))
# TODO: perform * explicitly
function *(A::HarmonicArray, C::AbstractMatrix)
    m, n, k = size(A.values)
    nc = size(C,2)
    T = promote_type(eltype(A.values),eltype(C))
    vals = Array{T,3}(undef,m,nc,k)
    [vals[:,:,i] = view(A.values,1:m,1:n,i)*C for i in 1:k]
    return HarmonicArray(vals, A.period; nperiod = A.nperiod)
    #*(A, HarmonicArray(C, A.period))
end
function *(A::AbstractMatrix, C::HarmonicArray) 
    m, n, k = size(C.values)
    ma = size(A,1)
    return HarmonicArray(reshape(A*reshape(C.values,m,n*k),ma,n,k), C.period; nperiod = C.nperiod)
end
*(A::HarmonicArray, C::Real) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)
*(C::Real, A::HarmonicArray) = HarmonicArray(C*A.values, A.period; nperiod = A.nperiod)
/(A::HarmonicArray, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::HarmonicArray) = J.λ*A
*(A::HarmonicArray, J::UniformScaling{<:Real}) = A*J.λ
*(A::HarmonicArray, C::PeriodicFunctionMatrix) = *(convert(PeriodicFunctionMatrix,A), C)
*(A::PeriodicFunctionMatrix, C::HarmonicArray) = *(A, convert(PeriodicFunctionMatrix,C))
function pmmultraddsym(A::HarmonicArray,B::HarmonicArray,C::HarmonicArray, (α,β) = (true, true))
    convert(HarmonicArray,pmmultraddsym(convert(PeriodicFunctionMatrix,A), convert(PeriodicFunctionMatrix,B), convert(PeriodicFunctionMatrix,C), (α,β)))
end
function pmmultraddsym(A::AbstractMatrix,B::HarmonicArray,C::HarmonicArray, (α,β) = (true, true))
    convert(HarmonicArray,pmmultraddsym(PeriodicFunctionMatrix(A, B.period), convert(PeriodicFunctionMatrix,B), convert(PeriodicFunctionMatrix,C), (α,β)))
end




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

function horzcat(A::HarmonicArray, B::HarmonicArray)
    if A.period == B.period && A.nperiod == B.nperiod
       m, n, la = size(A.values)
       mb, nb, lb = size(B.values)
       m == mb || throw(DimensionMismatch("A and B must have the same number of rows"))
       T = promote_type(eltype(A),eltype(B))
       lmax = max(la,lb)
       Ahr = zeros(Complex{T},m,n+nb,lmax)
       copyto!(view(Ahr,1:m,1:n,1:la), view(A.values,1:m,1:n,1:la))
       copyto!(view(Ahr,1:m,n+1:n+nb,1:lb), view(B.values,1:m,1:nb,1:lb))
       return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod) 
    else
       #TODO: fix different numbers of subperiods 
       convert(HarmonicArray,[convert(PeriodicFunctionMatrix,A) convert(PeriodicFunctionMatrix,B)])
    end
end
hcat(A::HarmonicArray, B::HarmonicArray) = horzcat(A,B)
hcat(A::HarmonicArray, C::AbstractMatrix) = horzcat(A, HarmonicArray(C, A.period))
hcat(A::AbstractMatrix, C::HarmonicArray) = horzcat(HarmonicArray(A, C.period), C)
horzcat(A::HarmonicArray, C::AbstractMatrix) = horzcat(A, HarmonicArray(C, A.period))
horzcat(A::AbstractMatrix, C::HarmonicArray) = horzcat(HarmonicArray(A, C.period), C)


function vertcat(A::HarmonicArray, B::HarmonicArray)
    if A.period == B.period && A.nperiod == B.nperiod
       m, n, la = size(A.values)
       mb, nb, lb = size(B.values)
       n == nb || throw(DimensionMismatch("A and B must have the same number of columns"))
       T = promote_type(eltype(A),eltype(B))
       lmax = max(la,lb)
       Ahr = zeros(Complex{T},m+mb,n,lmax)
       copyto!(view(Ahr,1:m,1:n,1:la), view(A.values,1:m,1:n,1:la))
       copyto!(view(Ahr,m+1:m+mb,1:n,1:la), view(B.values,1:mb,1:nb,1:lb))
       return HarmonicArray{:c,T}(Ahr, A.period, nperiod = A.nperiod) 
    else
       #TODO: fix different numbers of subperiods 
       convert(HarmonicArray,[convert(PeriodicFunctionMatrix,A); convert(PeriodicFunctionMatrix,B)])
    end
end
vcat(A::HarmonicArray, B::HarmonicArray) = vertcat(A,B)
vcat(A::HarmonicArray, C::AbstractMatrix) = vertcat(A, HarmonicArray(C, A.period))
vcat(A::AbstractMatrix, C::HarmonicArray) = vertcat(HarmonicArray(A, C.period), C)
vertcat(A::HarmonicArray, C::AbstractMatrix) = vertcat(A, HarmonicArray(C, A.period))
vertcat(A::AbstractMatrix, C::HarmonicArray) = vertcat(HarmonicArray(A, C.period), C)


function blockdiag(A::HarmonicArray, B::HarmonicArray)
    if A.period == B.period && A.nperiod == B.nperiod
        ma, na, la = size(A.values)
        mb, nb, lb = size(B.values)
        T = promote_type(eltype(A),eltype(B))
        lmax = max(la,lb)
        Ahr = zeros(Complex{T},ma+mb,na+nb,lmax)
        copyto!(view(Ahr,1:ma,1:na,1:la),A.values) 
        copyto!(view(Ahr,ma+1:ma+mb,na+1:na+nb,1:lb),B.values) 
        return HarmonicArray{:c,real(T)}(Ahr, A.period, nperiod = A.nperiod) 
      else
        #TODO: fix different numbers of subperiods 
        convert(HarmonicArray,blockdiag(convert(PeriodicFunctionMatrix,A),convert(PeriodicFunctionMatrix,B)))
     end
 end



#FourierFunctionMatrices
derivative(A::FourierFunctionMatrix{:c,T}) where {T} = FourierFunctionMatrix{:c,T}(differentiate(A.M), A.period, A.nperiod)
LinearAlgebra.inv(A::FourierFunctionMatrix) = FourierFunctionMatrix(inv(A.M), A.period; nperiod = A.nperiod)
LinearAlgebra.transpose(A::FourierFunctionMatrix{:c,T}) where {T}  = FourierFunctionMatrix{:c,T}(transpose(A.M), A.period, A.nperiod)
LinearAlgebra.adjoint(A::FourierFunctionMatrix) = transpose(A)
function LinearAlgebra.tr(V::Fun)
    typeof(size(space(V))) == Tuple{} && (return V)
    n, m = size(space(V))
    if n ≠ m
        throw(DimensionMismatch("space $(space(V)) is not square"))
    end
    a = Array(V)
    temp = a[1,1]
    for i = 2:n
        temp += a[i,i]
    end
    return temp
end
LinearAlgebra.tr(A::FourierFunctionMatrix) = FourierFunctionMatrix(tr(A.M), A.period; nperiod = A.nperiod)
function trace(A::FourierFunctionMatrix; K = 128) 
    isconstant(A) && (return tr(tpmeval(A, 0)))
    Δ = A.period/A.nperiod/K
    ts = Δ/2
    tt = zero(eltype(Δ))
    for i = 1:K
        tt += tr(tpmeval(A, ts))*Δ
        ts += Δ
    end 
    return tt*A.nperiod/A.period
end
function LinearAlgebra.opnorm(A::FourierFunctionMatrix, p::Union{Real, Missing} = missing) 
    return convert(FourierFunctionMatrix,opnorm(convert(PeriodicFunctionMatrix,A),p))
end
function norm(A::FourierFunctionMatrix, p::Real = 2; K = 128) 
    isconstant(A) && (return norm(tpmeval(A, 0)))
    nrm = zero(eltype(A))
    Δ = A.period/A.nperiod/K
    ts = Δ/2
    if p == 2
       for i = 1:K
           nrm += norm(tpmeval(A, ts))^2*Δ
           ts += Δ
       end 
       return sqrt(nrm*A.nperiod)
    elseif isinf(p)
        for i = 1:K
            nrm = max(nrm,norm(tpmeval(A, ts)))
            ts += Δ
        end 
        return nrm
    elseif p == 1    
        for i = 1:K
            nrm += norm(tpmeval(A, ts))*Δ
            ts += Δ
        end 
        return nrm*A.nperiod
    else
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
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
+(A::FourierFunctionMatrix, C::PeriodicFunctionMatrix) = +(convert(PeriodicFunctionMatrix,A), C)
+(A::PeriodicFunctionMatrix, C::FourierFunctionMatrix) = +(A, convert(PeriodicFunctionMatrix,C))
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
-(A::FourierFunctionMatrix, C::PeriodicFunctionMatrix) = +(A,-C)
-(A::PeriodicFunctionMatrix, C::FourierFunctionMatrix) = +(A, -C)


function *(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(A.M*B.M, A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,convert(PeriodicFunctionMatrix,A) * convert(PeriodicFunctionMatrix,B))
end
*(A::FourierFunctionMatrix, C::AbstractMatrix) = *(A, FourierFunctionMatrix(C, A.period))
*(A::AbstractMatrix, C::FourierFunctionMatrix) = *(FourierFunctionMatrix(A, C.period), C)
*(A::FourierFunctionMatrix, C::Real) = FourierFunctionMatrix(C*A.M, A.period)
*(C::Real, A::FourierFunctionMatrix) = FourierFunctionMatrix(C*A.M, A.period)
/(A::FourierFunctionMatrix, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::FourierFunctionMatrix) = J.λ*A
*(A::FourierFunctionMatrix, J::UniformScaling{<:Real}) = A*J.λ
*(A::FourierFunctionMatrix, C::PeriodicFunctionMatrix) = *(convert(PeriodicFunctionMatrix,A), C)
*(A::PeriodicFunctionMatrix, C::FourierFunctionMatrix) = *(A, convert(PeriodicFunctionMatrix,C))


function horzcat(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(Fun(t->[A.M(t) B.M(t)]), A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,[convert(PeriodicFunctionMatrix,A) convert(PeriodicFunctionMatrix,B)])
end
hcat(A::FourierFunctionMatrix, B::FourierFunctionMatrix) = horzcat(A,B)
hcat(A::FourierFunctionMatrix, C::AbstractMatrix) = horzcat(A, FourierFunctionMatrix(C, A.period))
hcat(A::AbstractMatrix, C::FourierFunctionMatrix) = horzcat(FourierFunctionMatrix(A, C.period), C)
horzcat(A::FourierFunctionMatrix, C::AbstractMatrix) = horzcat(A, FourierFunctionMatrix(C, A.period))
horzcat(A::AbstractMatrix, C::FourierFunctionMatrix) = horzcat(FourierFunctionMatrix(A, C.period), C)


function vertcat(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(Fun(t->[A.M(t); B.M(t)]), A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,[convert(PeriodicFunctionMatrix,A); convert(PeriodicFunctionMatrix,B)])
end
vcat(A::FourierFunctionMatrix, B::FourierFunctionMatrix) = vertcat(A,B)
vcat(A::FourierFunctionMatrix, C::AbstractMatrix) = vertcat(A, FourierFunctionMatrix(C, A.period))
vcat(A::AbstractMatrix, C::FourierFunctionMatrix) = vertcat(FourierFunctionMatrix(A, C.period), C)
vertcat(A::FourierFunctionMatrix, C::AbstractMatrix) = vertcat(A, FourierFunctionMatrix(C, A.period))
vertcat(A::AbstractMatrix, C::FourierFunctionMatrix) = vertcat(FourierFunctionMatrix(A, C.period), C)


function blockdiag(A::FourierFunctionMatrix, B::FourierFunctionMatrix)
    A.period == B.period && A.nperiod == B.nperiod && (return FourierFunctionMatrix(Fun(t->DescriptorSystems.blockdiag(A.M(t), B.M(t))), A.period; nperiod = A.nperiod))
    convert(FourierFunctionMatrix,DescriptorSystems.blockdiag(convert(PeriodicFunctionMatrix,A), convert(PeriodicFunctionMatrix,B)))
end



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
function derivative(A::PeriodicTimeSeriesMatrix{:c,T}; kwargs...) where {T}
    N = length(A)
    #tvmdereval(A, (0:N-1)*A.period/A.nperiod/N)
    #PeriodicTimeSeriesMatrix{:c,T}(tvmeval(derivative(convert(HarmonicArray,A)), collect((0:N-1)*A.period/A.nperiod/N)), A.period, A.nperiod)
    convert(PeriodicTimeSeriesMatrix,derivative(convert(PeriodicFunctionMatrix,A);kwargs...); ns = length(A))
end
LinearAlgebra.inv(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix(inv.(A.values), A.period; nperiod = A.nperiod)
LinearAlgebra.transpose(A::PeriodicTimeSeriesMatrix{:c,T}) where {T} = 
    PeriodicTimeSeriesMatrix{:c,T}([copy(transpose(A.values[i])) for i in 1:length(A)], A.period, A.nperiod)
LinearAlgebra.adjoint(A::PeriodicTimeSeriesMatrix) = transpose(A)
#LinearAlgebra.tr(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix(tr.(A.values), A.period; nperiod = A.nperiod)
LinearAlgebra.tr(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix([[tr(A.values[i])] for i in 1:length(A)], A.period; nperiod = A.nperiod)
function trace(A::PeriodicTimeSeriesMatrix) 
    K = length(A)
    K == 0 && (return zeros(eltype(A)))
    tt = tr(A.values[1])
    for i = 2:K
        tt += tr(A.values[i])
    end 
    return tt*A.nperiod/K
end
LinearAlgebra.eigvals(A::PeriodicTimeSeriesMatrix) = [eigvals(A.values[i]) for i in 1:length(A)]
function LinearAlgebra.opnorm(A::PeriodicTimeSeriesMatrix, p::Union{Real,Missing} = missing)
    if ismissing(p)
        return PeriodicTimeSeriesMatrix([[norm(A.values[i])] for i in 1:length(A)], A.period; nperiod = A.nperiod)
    else
        return PeriodicTimeSeriesMatrix([[opnorm(A.values[i],p)] for i in 1:length(A)], A.period; nperiod = A.nperiod)
    end
end
function norm(A::PeriodicTimeSeriesMatrix, p::Real = 2) 
    K = length(A)
    K == 0 && (return zeros(eltype(A)))
    nrm = zero(eltype(A))
    if p == 2 
       for i = 1:K
           nrm += norm(A.values[i])^2
       end 
       return sqrt(nrm*A.period/K)
    elseif isinf(p) 
        for i = 1:K
            nrm = max(nrm,norm(A.values[i]))
        end 
        return nrm
    elseif p == 1 
        for i = 1:K
            nrm += norm(A.values[i])
        end 
        return nrm*A.period/K
    else 
       throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end
function +(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) &&
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[i] for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([B.values[i]+A.values[1] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[1] for i in 1:length(A)], A.period, A.nperiod))
    if A.period == B.period 
       nperiod = gcd(A.nperiod,B.nperiod)
       ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B)),nperiod)
       Δ = A.period/nperiod/ns
       δ = Δ/2
       return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,(i-1)*Δ+δ)+tpmeval(B,(i-1)*Δ+δ) for i in 1:ns], A.period, nperiod) 
    else       
       Tsub = A.period/A.nperiod
       Tsub ≈ B.period/B.nperiod || error("periods or subperiods must be equal for addition")
       nperiod = lcm(A.nperiod,B.nperiod)
       period = Tsub*nperiod
       ns = lcm(length(A),length(B))
       Δ = Tsub/ns
       δ = Δ/2
       return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,(i-1)*Δ+δ)+tpmeval(B,(i-1)*Δ+δ) for i in 1:ns], period, nperiod)   
    end     
end
+(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = +(A, PeriodicTimeSeriesMatrix(C, A.period))
+(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = +(PeriodicTimeSeriesMatrix(A, C.period), C)
+(A::PeriodicTimeSeriesMatrix, C::PeriodicFunctionMatrix) = +(convert(PeriodicFunctionMatrix,A;method="constant"), C)
+(A::PeriodicFunctionMatrix, C::PeriodicTimeSeriesMatrix) = +(A, convert(PeriodicFunctionMatrix,C;method="constant"))
-(A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix{:c,eltype(A)}([-A.values[i] for i in 1:length(A)], A.period, A.nperiod)
-(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix) = +(A,-B)
-(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = +(A, -C)
-(A::PeriodicTimeSeriesMatrix, C::PeriodicFunctionMatrix) = +(A,-C)
-(A::PeriodicFunctionMatrix, C::PeriodicTimeSeriesMatrix) = +(A, -C)

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
    if A.period == B.period 
       nperiod = gcd(A.nperiod,B.nperiod)
       ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B)),nperiod)
       Δ = A.period/nperiod/ns
       δ = Δ/2
       return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,(i-1)*Δ+δ)*tpmeval(B,(i-1)*Δ+δ) for i in 1:ns], A.period, nperiod) 
    else          
       Tsub = A.period/A.nperiod
       Tsub ≈ B.period/B.nperiod || error("periods or subperiods must be equal for multiplication")
       nperiod = lcm(A.nperiod,B.nperiod)
       period = Tsub*nperiod
       ns = lcm(length(A),length(B))
       Δ = Tsub/ns
       δ = Δ/2
       return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,(i-1)*Δ+δ)*tpmeval(B,(i-1)*Δ+δ) for i in 1:ns], period, nperiod)   
    end     
end
*(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = *(A, PeriodicTimeSeriesMatrix(C, A.period))
*(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = *(PeriodicTimeSeriesMatrix(A, C.period), C)
*(A::PeriodicTimeSeriesMatrix, C::Real) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([A.values[i]*C for i in 1:length(A)], A.period, A.nperiod)
*(C::Real, A::PeriodicTimeSeriesMatrix) = PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(C))}([C*A.values[i] for i in 1:length(A)], A.period, A.nperiod)
*(A::PeriodicTimeSeriesMatrix, C::PeriodicFunctionMatrix) = *(convert(PeriodicFunctionMatrix,A;method="constant"), C)
*(A::PeriodicFunctionMatrix, C::PeriodicTimeSeriesMatrix) = *(A, convert(PeriodicFunctionMatrix,C;method="constant"))
/(A::PeriodicTimeSeriesMatrix, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::PeriodicTimeSeriesMatrix) = J.λ*A
*(A::PeriodicTimeSeriesMatrix, J::UniformScaling{<:Real}) = A*J.λ


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

function horzcat(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && 
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i] B.values[i]] for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[1] B.values[i]] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i] B.values[1]] for i in 1:length(A)], A.period, A.nperiod))
    if A.period == B.period 
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B)),nperiod)
        Δ = A.period/nperiod/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,(i-1)*Δ+δ) tpmeval(B,(i-1)*Δ+δ)] for i in 1:ns], A.period, nperiod) 
    else          
        Tsub = A.period/A.nperiod
        Tsub ≈ B.period/B.nperiod || error("periods or subperiods must be equal for horizontal concatenation")
        nperiod = lcm(A.nperiod,B.nperiod)
        period = Tsub*nperiod
        ns = lcm(length(A),length(B))
        Δ = Tsub/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,(i-1)*Δ+δ) tpmeval(B,(i-1)*Δ+δ)] for i in 1:ns], period, nperiod)   
    end     
end
hcat(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix) = horzcat(A,B)
hcat(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = horzcat(A, PeriodicTimeSeriesMatrix(C, A.period))
hcat(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = horzcat(PeriodicTimeSeriesMatrix(A, C.period), C)
horzcat(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = horzcat(A, PeriodicTimeSeriesMatrix(C, A.period))
horzcat(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = horzcat(PeriodicTimeSeriesMatrix(A, C.period), C)


function vertcat(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && 
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i]; B.values[i]] for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[1]; B.values[i]] for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i]; B.values[1]] for i in 1:length(A)], A.period, A.nperiod))
    if A.period == B.period 
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B)),nperiod)
        Δ = A.period/nperiod/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,(i-1)*Δ+δ); tpmeval(B,(i-1)*Δ+δ)] for i in 1:ns], A.period, nperiod) 
    else          
        Tsub = A.period/A.nperiod
        Tsub ≈ B.period/B.nperiod || error("periods or subperiods must be equal for vertical concatenation")
        nperiod = lcm(A.nperiod,B.nperiod)
        period = Tsub*nperiod
        ns = lcm(length(A),length(B))
        Δ = Tsub/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,(i-1)*Δ+δ); tpmeval(B,(i-1)*Δ+δ)] for i in 1:ns], period, nperiod)   
    end     
end
vcat(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix) = vertcat(A,B)
vcat(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = vertcat(A, PeriodicTimeSeriesMatrix(C, A.period))
vcat(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = vertcat(PeriodicTimeSeriesMatrix(A, C.period), C)
vertcat(A::PeriodicTimeSeriesMatrix, C::AbstractMatrix) = vertcat(A, PeriodicTimeSeriesMatrix(C, A.period))
vertcat(A::AbstractMatrix, C::PeriodicTimeSeriesMatrix) = vertcat(PeriodicTimeSeriesMatrix(A, C.period), C)

function blockdiag(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && 
        (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[i],B.values[i]) for i in 1:length(A)], A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[1],B.values[i]) for i in 1:length(B)], B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[i],B.values[1]) for i in 1:length(A)], A.period, A.nperiod))
    if A.period == B.period 
        nperiod = gcd(A.nperiod,B.nperiod)
        ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B)),nperiod)
        Δ = A.period/nperiod/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(tpmeval(A,(i-1)*Δ+δ), tpmeval(B,(i-1)*Δ+δ)) for i in 1:ns], A.period, nperiod) 
    else          
        Tsub = A.period/A.nperiod
        Tsub ≈ B.period/B.nperiod || error("periods or subperiods must be equal for block-diagonal appending")
        nperiod = lcm(A.nperiod,B.nperiod)
        period = Tsub*nperiod
        ns = lcm(length(A),length(B))
        Δ = Tsub/ns
        δ = Δ/2
        return PeriodicTimeSeriesMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(tpmeval(A,(i-1)*Δ+δ),tpmeval(B,(i-1)*Δ+δ)) for i in 1:ns], period, nperiod)   
    end     
end



# Operations with periodic switching matrices
# function derivative(A::PeriodicSwitchingMatrix{:c,T}) where {T}
#     PeriodicSwitchingMatrix{:c,T}([zeros(T,size(A,1),size(A,2)) for i in 1:length(A)], A.ts, A.period, A.nperiod)
# end
function derivative(A::PeriodicSwitchingMatrix{:c,T}; kwargs...) where {T}
    # N = length(A)
    # #tvmdereval(A, (0:N-1)*A.period/A.nperiod/N)
    # PeriodicSwitchingMatrix{:c,T}(tvmeval(derivative(convert(HarmonicArray,A)), collect((0:N-1)*A.period/A.nperiod/N)), A.ts, A.period, A.nperiod)
    # #PeriodicSwitchingMatrix{:c,T}(tvmeval(derivative(convert(PeriodicFunctionMatrix,A)), collect((0:N-1)*A.period/A.nperiod/N)), A.ts, A.period, A.nperiod)
    convert(PeriodicSwitchingMatrix,derivative(convert(PeriodicFunctionMatrix,A);kwargs...))
end

LinearAlgebra.inv(A::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix(inv.(A.values), A.ts, A.period; nperiod = A.nperiod)
LinearAlgebra.transpose(A::PeriodicSwitchingMatrix{:c,T}) where {T} = 
    PeriodicSwitchingMatrix{:c,T}([copy(transpose(A.values[i])) for i in 1:length(A)], A.ts, A.period, A.nperiod)
LinearAlgebra.adjoint(A::PeriodicSwitchingMatrix) = transpose(A)
#LinearAlgebra.tr(A::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix(tr.(A.values), A.period; nperiod = A.nperiod)
LinearAlgebra.tr(A::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix([[tr(A.values[i])] for i in 1:length(A)], A.ts, A.period; nperiod = A.nperiod)
function trace(A::PeriodicSwitchingMatrix) 
    K = length(A)
    K == 0 && (return zeros(eltype(A)))
    tt = tr(A.values[K])*(A.period/A.nperiod-A.ts[K])
    for i = 1:K-1
        tt += tr(A.values[i])*(A.ts[i+1]-A.ts[i])
    end 
    return tt*A.nperiod/A.period
end

LinearAlgebra.eigvals(A::PeriodicSwitchingMatrix) = [eigvals(A.values[i]) for i in 1:length(A)]
function LinearAlgebra.opnorm(A::PeriodicSwitchingMatrix, p::Union{Real,Missing} = missing)
    if ismissing(p)
        return PeriodicSwitchingMatrix([[norm(A.values[i])] for i in 1:length(A)], A.ts, A.period; nperiod = A.nperiod)
    else
        return PeriodicSwitchingMatrix([[opnorm(A.values[i],p)] for i in 1:length(A)], A.ts, A.period; nperiod = A.nperiod)
    end
end
function norm(A::PeriodicSwitchingMatrix, p::Real = 2) 
    K = length(A)
    K == 0 && (return zeros(eltype(A)))
    Δ = A.period/A.nperiod/K
    ts = zero(eltype(Δ))
    if p == 2
        nrm = norm(A.values[K])^2*(A.period/A.nperiod-A.ts[K])
        for i = 1:K-1
            nrm += norm(A.values[i])^2*(A.ts[i+1]-A.ts[i])
        end 
        return sqrt(nrm*A.nperiod)
    elseif isinf(p)
        nrm = zero(eltype(A))
        for i = 1:K-1
            nrm = max(nrm,norm(A.values[i]))
        end 
        return nrm
    elseif p == 1    
        nrm = norm(A.values[K])*(A.period/A.nperiod-A.ts[K])
        for i = 1:K-1
            nrm += norm(A.values[i])*(A.ts[i+1]-A.ts[i])
        end 
        return nrm*A.nperiod
    else
        throw(ArgumentError("only p-norms for p = 1, 2, or Inf are supported"))
    end
end
function +(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && A.ts == B.ts &&
        (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[i] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([B.values[i]+A.values[1] for i in 1:length(B)], B.ts, B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]+B.values[1] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for addition")
    if A.nperiod == B.nperiod
        ts = unique(sort([A.ts;B.ts]))
    else
        ts = unique(sort([vcat([(i-1)*A.period/A.nperiod .+ A.ts for i in 1:A.nperiod]...);
                          vcat([(i-1)*B.period/B.nperiod .+ B.ts for i in 1:B.nperiod]...)]))
    end
    return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,ts[i])+tpmeval(B,ts[i]) for i in 1:length(ts)], ts, A.period, gcd(A.nperiod,B.nperiod))
end
+(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = +(A, PeriodicSwitchingMatrix([C], [0], A.period))
+(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = +(PeriodicSwitchingMatrix([A], [0], C.period), C)
+(A::PeriodicSwitchingMatrix, C::PeriodicFunctionMatrix) = +(convert(PeriodicFunctionMatrix,A), C)
+(A::PeriodicFunctionMatrix, C::PeriodicSwitchingMatrix) = +(A, convert(PeriodicFunctionMatrix,C))
-(A::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix{:c,eltype(A)}([-A.values[i] for i in 1:length(A)], A.ts, A.period, A.nperiod)
-(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix) = +(A,-B)
-(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = +(A,-C)
-(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = +(A, -C)
-(A::PeriodicSwitchingMatrix, C::PeriodicFunctionMatrix) = +(A,-C)
-(A::PeriodicFunctionMatrix, C::PeriodicSwitchingMatrix) = +(A, -C)

function (+)(A::PeriodicSwitchingMatrix, J::UniformScaling{<:Real}) 
    n = size(A,1)
    n == size(A,2) || throw(DimensionMismatch("A must be square"))
    A+Matrix(J(n))
end
(+)(J::UniformScaling{<:Real}, A::PeriodicSwitchingMatrix) = +(A,J)
(-)(A::PeriodicSwitchingMatrix, J::UniformScaling{<:Real}) = +(A,-J)
(-)(J::UniformScaling{<:Real}, A::PeriodicSwitchingMatrix) = +(-A,J)

function *(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && A.ts == B.ts &&
        (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]*B.values[i] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[1]*B.values[i] for i in 1:length(B)], B.ts, B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([A.values[i]*B.values[1] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for multiplication")
    if A.nperiod == B.nperiod
        ts = unique(sort([A.ts;B.ts]))
    else
        ts = unique(sort([vcat([(i-1)*A.period/A.nperiod .+ A.ts for i in 1:A.nperiod]...);
                          vcat([(i-1)*B.period/B.nperiod .+ B.ts for i in 1:B.nperiod]...)]))
    end
    return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([tpmeval(A,ts[i])*tpmeval(B,ts[i]) for i in 1:length(ts)], ts, A.period, gcd(A.nperiod,B.nperiod))
   end
*(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(C))}([A.values[i]*C for i in 1:length(A)], A.ts, A.period, A.nperiod)
*(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(C))}([A*C.values[i] for i in 1:length(C)], C.ts, C.period, C.nperiod)
*(A::PeriodicSwitchingMatrix, C::Real) = PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(C))}([A.values[i]*C for i in 1:length(A)], A.ts, A.period, A.nperiod)
*(C::Real, A::PeriodicSwitchingMatrix) = PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(C))}([C*A.values[i] for i in 1:length(A)], A.ts, A.period, A.nperiod)
*(A::PeriodicSwitchingMatrix, C::PeriodicFunctionMatrix) = *(convert(PeriodicFunctionMatrix,A), C)
*(A::PeriodicFunctionMatrix, C::PeriodicSwitchingMatrix) = *(A, convert(PeriodicFunctionMatrix,C))
/(A::PeriodicSwitchingMatrix, C::Real) = *(A, 1/C)
*(J::UniformScaling{<:Real}, A::PeriodicSwitchingMatrix) = J.λ*A
*(A::PeriodicSwitchingMatrix, J::UniformScaling{<:Real}) = A*J.λ

Base.iszero(A::PeriodicSwitchingMatrix) = iszero(A.values)
LinearAlgebra.issymmetric(A::PeriodicSwitchingMatrix) = all(issymmetric.(A.values))
function ==(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    isconstant(A) && isconstant(B) && (return isequal(A.values, B.values))
    isequal(A.values, B.values) &&  (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && A.ts == B.ts
end
function Base.isapprox(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix; rtol::Real = sqrt(eps(Float64)), atol::Real = 0)
    isconstant(A) && isconstant(B) && (return isapprox(A.values, B.values; rtol, atol))
    isapprox(A.values, B.values; rtol, atol) && (A.period == B.period || A.period*B.nperiod == B.period*A.nperiod) && A.ts == B.ts
end
function Base.isapprox(A::PeriodicSwitchingMatrix, J::UniformScaling{<:Real}; kwargs...)
    all([isapprox(A.values[i], J; kwargs...) for i in 1:length(A.values)])
end
Base.isapprox(J::UniformScaling{<:Real}, A::PeriodicSwitchingMatrix; kwargs...) = isapprox(A, J; kwargs...)

function horzcat(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && A.ts == B.ts &&
        (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i] B.values[i]] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[1] B.values[i]] for i in 1:length(B)], B.ts, B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i] B.values[1]] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for horizontal concatenation")
    if A.nperiod == B.nperiod
        ts = unique(sort([A.ts;B.ts]))
    else
        ts = unique(sort([vcat([(i-1)*A.period/A.nperiod .+ A.ts for i in 1:A.nperiod]...);
                          vcat([(i-1)*B.period/B.nperiod .+ B.ts for i in 1:B.nperiod]...)]))
    end
    return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,ts[i]) tpmeval(B,ts[i])] for i in 1:length(ts)], ts, A.period, gcd(A.nperiod,B.nperiod))
end
hcat(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix) = horzcat(A,B)
hcat(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = horzcat(A, PeriodicSwitchingMatrix(C, A.period))
hcat(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = horzcat(PeriodicSwitchingMatrix(A, C.period), C)
horzcat(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = horzcat(A, PeriodicSwitchingMatrix(C, A.period))
horzcat(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = horzcat(PeriodicSwitchingMatrix(A, C.period), C)

function vertcat(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && A.ts == B.ts &&
        (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i]; B.values[i]] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[1]; B.values[i]] for i in 1:length(B)], B.ts, B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[A.values[i]; B.values[1]] for i in 1:length(A)], A.ts, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for vertical concatenation")
    if A.nperiod == B.nperiod
        ts = unique(sort([A.ts;B.ts]))
    else
        ts = unique(sort([vcat([(i-1)*A.period/A.nperiod .+ A.ts for i in 1:A.nperiod]...);
                          vcat([(i-1)*B.period/B.nperiod .+ B.ts for i in 1:B.nperiod]...)]))
    end
    return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([[tpmeval(A,ts[i]); tpmeval(B,ts[i])] for i in 1:length(ts)], ts, A.period, gcd(A.nperiod,B.nperiod))
end
vcat(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix) = vertcat(A,B)
vcat(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = vertcat(A, PeriodicSwitchingMatrix(C, A.period))
vcat(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = vertcat(PeriodicSwitchingMatrix(A, C.period), C)
vertcat(A::PeriodicSwitchingMatrix, C::AbstractMatrix) = vertcat(A, PeriodicSwitchingMatrix(C, A.period))
vertcat(A::AbstractMatrix, C::PeriodicSwitchingMatrix) = vertcat(PeriodicSwitchingMatrix(A, C.period), C)


function blockdiag(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix)
    A.period == B.period && A.nperiod == B.nperiod && length(A) == length(B) && A.ts == B.ts &&
        (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[i],B.values[i]) for i in 1:length(A)], A.ts, A.period, A.nperiod))
    isconstant(A) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[1], B.values[i]) for i in 1:length(B)], B.ts, B.period, B.nperiod))
    isconstant(B) && 
       (return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(A.values[i], B.values[1]) for i in 1:length(A)], A.ts, A.period, A.nperiod))
    A.period == B.period || error("periods must be equal for block-diagonal appending")
    if A.nperiod == B.nperiod
        ts = unique(sort([A.ts;B.ts]))
    else
        ts = unique(sort([vcat([(i-1)*A.period/A.nperiod .+ A.ts for i in 1:A.nperiod]...);
                          vcat([(i-1)*B.period/B.nperiod .+ B.ts for i in 1:B.nperiod]...)]))
    end
    return PeriodicSwitchingMatrix{:c,promote_type(eltype(A),eltype(B))}([DescriptorSystems.blockdiag(tpmeval(A,ts[i]), tpmeval(B,ts[i])) for i in 1:length(ts)], ts, A.period, gcd(A.nperiod,B.nperiod))
end

