for PM in (:PeriodicArray, :PeriodicMatrix)
    @eval begin
       function pdlyap(A::$PM, C::$PM; adj::Bool = true, stability_check = false) 
          A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
          period = promote_period(A, C)
          na = rationalize(period/A.period).num
          K = na*A.nperiod*A.dperiod
          X = pslyapd(A.M, C.M; adj, stability_check)
          p = lcm(length(A),length(C))
          return $PM(X, period; nperiod = div(K,p))
       end
       function pdlyap2(A::$PM, C::$PM, E::$PM; stability_check = false) 
          A.Ts  ≈ C.Ts ≈ E.Ts || error("A, C and E must have the same sampling time")
          period = promote_period(A, C, E)
          na = rationalize(period/A.period).num
          K = na*A.nperiod*A.dperiod
          X, Y = pslyapd2(A.M, C.M, E.M; stability_check)
          p = lcm(length(A),length(C),length(E))
          return $PM(X, period; nperiod = div(K,p)), $PM(Y, period; nperiod = div(K,p))
       end
    end
end
function pdlyap(A::PM, C::PM; kwargs...) where {PM <: SwitchingPeriodicMatrix}
   X = pdlyap(convert(PeriodicMatrix,A),convert(PeriodicMatrix,C); kwargs...)
   return convert(SwitchingPeriodicMatrix,X)
end
function pdlyap2(A::PM, C::PM, E::PM; kwargs...) where {PM <: SwitchingPeriodicMatrix}
   X, Y = pdlyap2(convert(PeriodicMatrix,A),convert(PeriodicMatrix,C),convert(PeriodicMatrix,E); kwargs...)
   return convert(SwitchingPeriodicMatrix,X), convert(SwitchingPeriodicMatrix,Y)
end
function pdlyap(A::PM, C::PM; kwargs...) where {PM <: SwitchingPeriodicArray}
   X = pdlyap(convert(PeriodicArray,A),convert(PeriodicArray,C); kwargs...)
   return convert(SwitchingPeriodicArray,X)
end
function pdlyap2(A::PM, C::PM, E::PM; kwargs...) where {PM <: SwitchingPeriodicArray}
   X, Y = pdlyap2(convert(PeriodicArray,A),convert(PeriodicArray,C),convert(PeriodicArray,E); kwargs...)
   return convert(SwitchingPeriodicArray,X), convert(SwitchingPeriodicArray,Y)
end


"""
    pdlyap(A, C; adj = true, stability_check = false) -> X

Solve the periodic discrete-time Lyapunov equation

    A'σXA + C = X  for adj = true 

or 

    AXA' + C = σX  for adj = false,
    
where `σ` is the forward shift operator `σX(i) = X(i+1)`. 

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrix `A` is employed [1].

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
pdlyap(A::PeriodicArray, C::PeriodicArray; adj::Bool = true) 
"""
    pdlyap2(A, C, E; stability_check = false) -> (X, Y)

Solve the pair of periodic discrete-time Lyapunov equations

    AXA' + C  = σX, 
    A'σYA + E = Y,
    
where `σ` is the forward shift operator `σX(i) = X(i+1)` and `σY(i) = Y(i+1)`. 

The periodic matrices `A`, `C` and `E` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` and `E` must be symmetric. The resulting symmetric periodic solutions `X` and `Y` have the period 
set to the least common commensurate period of `A`, `C` and `E` and the number of subperiods
is adjusted accordingly. 

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrix `A` is employed [1].

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
pdlyap2(A::PeriodicArray, C::PeriodicArray, E::PeriodicArray; stability_check = false)

for PM in (:PeriodicArray, :PeriodicMatrix, :SwitchingPeriodicMatrix, :SwitchingPeriodicArray)
   @eval begin
      function prdlyap(A::$PM, C::$PM; stability_check = false) 
         pdlyap(A, C; adj = true, stability_check)
      end
      function prdlyap(A::$PM, C::AbstractArray; stability_check = false)
               pdlyap(A, $PM(C, A.Ts; nperiod = 1);  adj = true, stability_check)
      end
      function pfdlyap(A::$PM, C::$PM; stability_check = false) 
         pdlyap(A, C; adj = false, stability_check)
      end
      function pfdlyap(A::$PM, C::AbstractArray; stability_check = false) 
         pdlyap(A, $PM(C, A.Ts; nperiod = 1); adj = false, stability_check)
      end
   end
end
"""
    prdlyap(A, C; stability_check = false) -> X

Solve the reverse-time periodic discrete-time Lyapunov equation

    A'σXA + C = X

where `σ` is the forward shift operator `σX(i) = X(i+1)`.                 

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 
"""
prdlyap(A::PeriodicArray, C::PeriodicArray) 
"""
    pfdlyap(A, C; stability_check = false) -> X

Solve the forward-time periodic discrete-time Lyapunov equation

    AXA' + C = σX

where `σ` is the forward shift operator `σX(i) = X(i+1)`.                 

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly.  

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 
"""
pfdlyap(A::PeriodicArray, C::PeriodicArray) 
"""
    pslyapd(A, C; adj = true, stability_check = false) -> X

Solve the periodic discrete-time Lyapunov equation.

For the square `n`-th order periodic matrices `A(i)`, `i = 1, ..., pa` and 
`C(i)`, `i = 1, ..., pc`  of periods `pa` and `pc`, respectively, 
the periodic solution `X(i)`, `i = 1, ..., p` of period `p = lcm(pa,pc)` of the 
periodic Lyapunov equation is computed:  

    A(i)'*X(i+1)*A(i) + C(i) = X(i), i = 1, ..., p     for `adj = true`; 

    A(i)*X(i)*A(i)' + C(i) = X(i+1), i = 1, ..., p     for `adj = false`.   

The periodic matrices `A` and `C` are stored in the `n×n×pa` and `n×n×pc` 3-dimensional 
arrays `A` and `C`, respectively, and `X` results as a `n×n×p` 3-dimensional array.  

Alternatively, the periodic matrices `A` and `C` can be stored in the  `pa`- and `pc`-dimensional
vectors of matrices `A` and `C`, respectively, and `X` results as a `p`-dimensional vector of matrices.

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrix `A` is employed [1].

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
function pslyapd(A::AbstractArray{T1, 3}, C::AbstractArray{T2, 3}; adj::Bool = true, stability_check = false) where {T1, T2}
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
      throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   p = lcm(pa,pc)

   T = promote_type(T1, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = T1 == T ? A : A1 = convert(Array{T,3},A)
   C1 = T2 == T ? C : C1 = convert(Array{T,3},C)

   # Reduce A to Schur form and transform C
   AS, Q, ev, KSCHUR = PeriodicMatrices.pschur(A1)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")
   
   #X = Q'*C*Q
   X = Array{T,3}(undef, n, n, p)

   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ia1 = mod(i,pa)+1

       X[:,:,i] = adj ? utqu(view(C1,:,:,ic),view(Q,:,:,ia)) : 
                        utqu(view(C1,:,:,ic),view(Q,:,:,ia1)) 
   end

   # solve A'σXA + C = X if adj = true or AXA' + C = σX if adj = false
   pdlyaps!(KSCHUR, AS, X; adj)

   #X <- Q*X*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       utqu!(view(X,:,:,i),view(Q,:,:,ia)')
   end
   return X
end
function pslyapd(A::AbstractVector{Matrix{T1}}, C::AbstractVector{Matrix{T2}}; adj::Bool = true, stability_check = false) where {T1, T2}
   pa = length(A) 
   pc = length(C)
   ma, na = size.(A,1), size.(A,2) 
   mc, nc = size.(C,1), size.(C,2) 
   p = lcm(pa,pc)
   all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   if adj
      all([LinearAlgebra.checksquare(C[mod(i-1,pc)+1]) == na[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   else
      all([LinearAlgebra.checksquare(C[mod(i-1,pc)+1]) == ma[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   end

   all([issymmetric(C[i]) for i in 1:pc]) || error("all C[i] must be symmetric matrices")

   n = maximum(na)

   T = promote_type(T1, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = zeros(T, n, n, pa)
   C1 = zeros(T, n, n, pc)
   [copyto!(view(A1,1:ma[i],1:na[i],i), T.(A[i])) for i in 1:pa]
   adj ? [copyto!(view(C1,1:nc[i],1:nc[i],i), T.(C[i])) for i in 1:pc] :
         [copyto!(view(C1,1:mc[i],1:mc[i],i), T.(C[i])) for i in 1:pc] 

   # Reduce A to Schur form and transform C
   AS, Q, ev, KSCHUR = PeriodicMatrices.pschur(A1)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")
   
   # if adj = true: X = Q'*C*Q; if adj = false: X = σQ'*C*σQ
   X = Array{T,3}(undef, n, n, p)

   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ia1 = mod(i,pa)+1

       X[:,:,i] = adj ? utqu(view(C1,:,:,ic),view(Q,:,:,ia)) : 
                        utqu(view(C1,:,:,ic),view(Q,:,:,ia1)) 
   end
   # solve A'σXA + C = X if adj = true or AXA' + C = σX if adj = false
   pdlyaps!(KSCHUR, AS, X; adj)

   #X <- Q*X*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       utqu!(view(X,:,:,i),view(Q,:,:,ia)')
   end
   return adj ? [X[1:na[mod(i-1,pa)+1],1:na[mod(i-1,pa)+1],i] for i in 1:p] :
                [X[1:na[mod(i-1,pa)+1],1:na[mod(i-1,pa)+1],i] for i in 1:p]  
end
"""
    pslyapd2(A, C, E; stability_check = false) -> (X, Y)

Solve a pair of periodic discrete-time Lyapunov equations.

For the square `n`-th order periodic matrices `A(i)`, `i = 1, ..., pa`, 
`C(i)`, `i = 1, ..., pc`, and `E(i)`, `i = 1, ..., pe` of periods `pa`, `pc` and `pe`, respectively, 
the periodic solutions `X(i)`, `i = 1, ..., p` and `Y(i)`, `i = 1, ..., p` 
of period `p = lcm(pa,pc,pe)` of the periodic Lyapunov equations are computed:  

    A(i)*X(i)*A(i)' + C(i) = X(i+1), i = 1, ..., p ,  
     
    A(i)'*Y(i+1)*A(i) + E(i) = Y(i), i = 1, ..., p . 

The periodic matrices `A`, `C` and `E` are stored in the `n×n×pa`, `n×n×pc` and `n×n×pe` 3-dimensional 
arrays `A`, `C` and `E`, respectively, and `X` and `Y` result as `n×n×p` 3-dimensional arrays.  

Alternatively, the periodic matrices `A`, `C` and `E` can be stored in the  `pa`-, `pc`- and `pe`-dimensional
vectors of matrices `A`, `C` and `E`, respectively, and `X` and `Y` result as `p`-dimensional vectors of matrices.

If `stability_check = true`, the stability of characteristic multipliers of `A` is checked and an error is issued
if any characteristic multiplier has modulus equal to or larger than one. 

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrix `A` is employed [1].

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
function pslyapd2(A::AbstractVector{Matrix{T1}}, C::AbstractVector{Matrix{T2}}, E::AbstractVector{Matrix{T3}}; stability_check = false) where {T1, T2, T3}
   pa = length(A) 
   pc = length(C)
   pe = length(E)
   ma, na = size.(A,1), size.(A,2) 
   mc, nc = size.(C,1), size.(C,2) 
   me, ne = size.(E,1), size.(E,2) 
   p = lcm(pa,pc,pe)
   all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   all([LinearAlgebra.checksquare(C[mod(i-1,pc)+1]) == ma[mod(i-1,pa)+1] for i in 1:p]) ||
        throw(DimensionMismatch("incompatible dimensions between A and C"))
   all([LinearAlgebra.checksquare(E[mod(i-1,pe)+1]) == na[mod(i-1,pa)+1] for i in 1:p]) ||
        throw(DimensionMismatch("incompatible dimensions between A and E"))


   all([issymmetric(C[i]) for i in 1:pc]) || error("all C[i] must be symmetric matrices")
   all([issymmetric(E[i]) for i in 1:pe]) || error("all E[i] must be symmetric matrices")

   n = maximum(na)

   T = promote_type(T1, T2, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = zeros(T, n, n, pa)
   C1 = zeros(T, n, n, pc)
   E1 = zeros(T, n, n, pe)
   [copyto!(view(A1,1:ma[i],1:na[i],i), T.(A[i])) for i in 1:pa]
   [copyto!(view(E1,1:ne[i],1:ne[i],i), T.(E[i])) for i in 1:pe] 
   [copyto!(view(C1,1:mc[i],1:mc[i],i), T.(C[i])) for i in 1:pc] 

   # Reduce A to Schur form and transform C
   AS, Q, ev, KSCHUR = PeriodicMatrices.pschur(A1)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")
   
   # Y = Q'*E*Q; X = σQ'*C*σQ
   X = Array{T,3}(undef, n, n, p)
   Y = Array{T,3}(undef, n, n, p)

   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ie = mod(i-1,pe)+1
       ia1 = mod(i,pa)+1

       X[:,:,i] = utqu(view(C1,:,:,ic),view(Q,:,:,ia1)) 
       Y[:,:,i] = utqu(view(E1,:,:,ie),view(Q,:,:,ia)) 
   end
   # solve A'σXA + C = X
   pdlyaps!(KSCHUR, AS, X; adj = false)
   # solve AXA' + E = σX 
   pdlyaps!(KSCHUR, AS, Y; adj = true)

   #X <- Q*X*Q', Y <- = Q*Y*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       utqu!(view(X,:,:,i),view(Q,:,:,ia)')
       utqu!(view(Y,:,:,i),view(Q,:,:,ia)')
   end
   return [X[1:na[mod(i-1,pa)+1],1:na[mod(i-1,pa)+1],i] for i in 1:p], [Y[1:na[mod(i-1,pa)+1],1:na[mod(i-1,pa)+1],i] for i in 1:p]                
end
function pslyapd2(A::AbstractArray{T1, 3}, C::AbstractArray{T2, 3}, E::AbstractArray{T3, 3}; stability_check = false) where {T1, T2, T3}
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   pe = size(E,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
           throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   (LinearAlgebra.checksquare(E[:,:,1]) == n && all([issymmetric(E[:,:,i]) for i in 1:pe])) ||
           throw(DimensionMismatch("all E[:,:,i] must be $n x $n symmetric matrices"))
   p = lcm(pa,pc,pe)

   T = promote_type(T1, T2, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = T1 == T ? A : A1 = convert(Array{T,3},A)
   C1 = T2 == T ? C : C1 = convert(Array{T,3},C)
   E1 = T2 == T ? E : E1 = convert(Array{T,3},E)

   # Reduce A to Schur form and transform C and E
   AS, Q, ev, KSCHUR = PeriodicMatrices.pschur(A1)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")
   
   # Y = Q'*E*Q; X = σQ'*C*σQ
   X = Array{T,3}(undef, n, n, p)
   Y = Array{T,3}(undef, n, n, p)
   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ie = mod(i-1,pe)+1
       ia1 = mod(i,pa)+1
       X[:,:,i] = utqu(view(C1,:,:,ic),view(Q,:,:,ia1)) 
       Y[:,:,i] = utqu(view(E1,:,:,ie),view(Q,:,:,ia)) 
   end

   # solve A'σXA + C = X
   pdlyaps!(KSCHUR, AS, X; adj = false)
   # solve AXA' + E = σX 
   pdlyaps!(KSCHUR, AS, Y; adj = true)

   #X <- Q*X*Q', Y <- = Q*Y*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       utqu!(view(X,:,:,i),view(Q,:,:,ia)')
       utqu!(view(Y,:,:,i),view(Q,:,:,ia)')
   end
   return X, Y                
end
function pslyapd!(X::AbstractArray{T, 3}, A::AbstractArray{T, 3}, C::AbstractArray{T, 3}, Xt::AbstractMatrix{T}, Q::AbstractArray{T, 3}; adj::Bool = true, stability_check = false) where {T}
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
           throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   p = lcm(pa,pc)

   # Reduce A to Schur form and transform C 
   ev, KSCHUR = PeriodicMatrices.pschur!(A,Q)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")

   for i = 1:p
       ic = mod(i-1,pc)+1
       if adj
          #  X[:,:,i] = utqu(view(C,:,:,ic),view(Q,:,:,ia)) 
          ia = mod(i-1,pa)+1
          mul!(Xt,view(Q,:,:,ia)',view(C,:,:,ic))
          muladdsym!(view(X,:,:,i),Xt,view(Q,:,:,ia),(0,1))
       else
          #  X[:,:,i] = utqu(view(C,:,:,ic),view(Q,:,:,ia1)) 
          ia1 = mod(i,pa)+1
          mul!(Xt,view(Q,:,:,ia1)',view(C,:,:,ic))
          muladdsym!(view(X,:,:,i),Xt,view(Q,:,:,ia1),(0,1))
       end
   end

   # solve A'σXA + C = X for adj = true or  AXA' + C = σX for adj = false 
   pdlyaps!(KSCHUR, A, X; adj)

   #X <- Q*X*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       #  utqu!(view(X,:,:,i),view(Q,:,:,ia)')
       mul!(Xt,view(X,:,:,i),view(Q,:,:,ia)')
       muladdsym!(view(X,:,:,i),view(Q,:,:,ia),Xt,(0,1))
   end
   return X               
end
function pslyapd2!(X::AbstractArray{T, 3}, Y::AbstractArray{T, 3}, A::AbstractArray{T, 3}, C::AbstractArray{T, 3}, E::AbstractArray{T, 3}, Xt::AbstractMatrix{T}, Q::AbstractArray{T, 3}, WORK, pschur_ws; stability_check = false) where {T}
   n = LinearAlgebra.checksquare(view(A,:,:,1))
   n == LinearAlgebra.checksquare(view(C,:,:,1)) ||
          throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   n == LinearAlgebra.checksquare(view(E,:,:,1)) ||
          throw(DimensionMismatch("all E[:,:,i] must be $n x $n symmetric matrices"))
   n == LinearAlgebra.checksquare(view(X,:,:,1)) ||
          throw(DimensionMismatch("all X[:,:,i] must be $n x $n matrices"))
   n == LinearAlgebra.checksquare(view(Y,:,:,1)) ||
          throw(DimensionMismatch("all Y[:,:,i] must be $n x $n matrices"))
   pa = size(A,3)
   pc = size(C,3)
   pe = size(E,3)
   # (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
   #         throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   # (LinearAlgebra.checksquare(E[:,:,1]) == n && all([issymmetric(E[:,:,i]) for i in 1:pe])) ||
   #         throw(DimensionMismatch("all E[:,:,i] must be $n x $n symmetric matrices"))
   p = lcm(pa,pc,pe)
   p == size(X,3) == size(Y,3) || throw(DimensionMismatch("incompatible third dimensions of X and Y with A, C, and E"))

   # Reduce A to Schur form and transform C and E
   ev, KSCHUR = PeriodicMatrices.pschur!(pschur_ws,A,Q)
   stability_check && maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")

   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ie = mod(i-1,pe)+1
       ia1 = mod(i,pa)+1
       #  X[:,:,i] = utqu(view(C,:,:,ic),view(Q,:,:,ia1)) 
       #  Y[:,:,i] = utqu(view(E,:,:,ie),view(Q,:,:,ia)) 
       mul!(Xt,view(Q,:,:,ia1)',view(C,:,:,ic))
       muladdsym!(view(X,:,:,i),Xt,view(Q,:,:,ia1),(0,1))
       mul!(Xt,view(Q,:,:,ia)',view(E,:,:,ie))
       muladdsym!(view(Y,:,:,i),Xt,view(Q,:,:,ia),(0,1))
   end

   # solve A'σXA + C = X and AYA' + E = σY 
   pdlyaps2!(KSCHUR, A, X, Y, WORK)
  
   #X <- Q*X*Q', Y <- = Q*Y*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       #  utqu!(view(X,:,:,i),view(Q,:,:,ia)')
       #  utqu!(view(Y,:,:,i),view(Q,:,:,ia)')
       mul!(Xt,view(X,:,:,i),view(Q,:,:,ia)')
       muladdsym!(view(X,:,:,i),view(Q,:,:,ia),Xt,(0,1))
       mul!(Xt,view(Y,:,:,i),view(Q,:,:,ia)')
       muladdsym!(view(Y,:,:,i),view(Q,:,:,ia),Xt,(0,1))
   end
   return nothing                
end

"""
     pslyapdkr(A, C; adj = true) -> X

Solve the periodic discrete-time Lyapunov matrix equation

      A'σXA + C = X, if adj = true,

or 

      A*X*A' + C =  σX, if adj = false, 

where `σ` is the forward shift operator `σX(i) = X(i+1)`.  
The periodic matrix `A` must not have characteristic multipliers on the unit circle.               
The periodic matrices `A` and `C` are either stored as 3-dimensional arrays or as
as vectors of matrices. 

The Kronecker product expansion of equations is employed and therefore 
this function is not recommended for large order matrices or large periods.
"""
function pslyapdkr(A::AbstractArray{T1, 3}, C::AbstractArray{T2, 3}; adj = true) where {T1, T2}
   m, n, pc = size(C)
   n == LinearAlgebra.checksquare(A[:,:,1]) 
   m == LinearAlgebra.checksquare(C[:,:,1]) 
   m == n  || throw(DimensionMismatch("A and C have incompatible dimensions"))
   pa = size(A,3)
   n2 = n*n
   p = lcm(pa,pc)
   N = p*n2
   R = zeros(promote_type(T1,T2), N, N)
   if adj
      copyto!(view(R,1:n2,N-n2+1:N),kron(A[:,:,pa]',A[:,:,pa]')) 
      i1 = n2+1; j1 = 1 
      for i = p-1:-1:1         
          ia = mod(i-1,pa)+1
          i2 = i1+n2-1
          j2 = j1+n2-1
          copyto!(view(R,i1:i2,j1:j2),kron(A[:,:,ia]',A[:,:,ia]')) 
          i1 = i2+1
          j1 = j2+1
      end
      indc = mod.(p-1:-1:0,pc).+1
      return reshape((I-R) \ (C[:,:,indc][:]), n, n, p)[:,:,indc] 
   else
      copyto!(view(R,1:n2,N-n2+1:N),kron(A[:,:,pa],A[:,:,pa])) 
      (i2, j2) = (n2+n2, n2)
      for i = 1:p-1
          i1 = i2-n2+1
          j1 = j2-n2+1
          ia = mod(i-1,pa)+1
          copyto!(view(R,i1:i2,j1:j2),kron(A[:,:,ia],A[:,:,ia])) 
          i2 += n2
          j2 += n2
      end
      indc = mod.(-1:p-2,pc).+1
      return reshape((I-R) \ (C[:,:,indc][:]), n, n, p)
   end
end
function pslyapdkr(A::AbstractVector{Matrix{T1}}, C::AbstractVector{Matrix{T2}}; adj = true) where {T1, T2}
   pa = length(A) 
   pc = length(C)
   ma, na = size.(A,1), size.(A,2) 
   p = lcm(pa,pc)
   all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   if adj
      all([LinearAlgebra.checksquare(C[mod(i-1,pc)+1]) == na[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   else
      all([LinearAlgebra.checksquare(C[mod(i-1,pc)+1]) == ma[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   end

   all([issymmetric(C[i]) for i in 1:pc]) || error("all C[i] must be symmetric matrices")
   
   T = promote_type(T1,T2)
   m2 = ma.^2
   n2 = na.^2
   mn = ma.*na
   p = lcm(pa,pc)
   N = adj ? sum(m2) : sum(n2)
   R = zeros(T, N, N)
   Y = zeros(T, N)
   
   if adj 
      copyto!(view(R,1:n2[pa],N-m2[pa]+1:N),kron(A[pa]',A[pa]')) 
      copyto!(view(Y,1:n2[pa]),C[pa][:]) 
      i1 = n2[pa]+1; j1 = 1 
      for i = p-1:-1:1         
          ia = mod(i-1,pa)+1
          ic = mod(i-1,pc)+1
          i2 = i1+n2[ia]-1
          j2 = j1+m2[ia]-1
          copyto!(view(R,i1:i2,j1:j2),kron(A[ia]',A[ia]')) 
          copyto!(view(Y,i1:i2),C[ic][:]) 
          i1 = i2+1
          j1 = j2+1
      end
      ldiv!(qr!(I-R),Y)
      z = Vector{Matrix{T}}(undef,0)
      i2 = N
      for i = 1:p 
          ia = mod(i-1,pa)+1
          ia1 = mod(i-2,pa)+1
          i1 = i2-m2[ia1]+1
          push!(z,reshape(view(Y,i1:i2),ma[ia1],ma[ia1]))
          i2 = i1-1
      end
      return z
   else
      copyto!(view(R,1:m2[pa],N-n2[pa]+1:N),kron(A[pa],A[pa])) 
      copyto!(view(Y,1:m2[pa]),C[pa][:]) 
      i1 = m2[pa]+1; j1 = 1 
      for i = 1:p-1
          ia = mod(i-1,pa)+1
          ic = mod(i-1,pc)+1
          i2 = i1+m2[ia]-1
          j2 = j1+n2[ia]-1
          copyto!(view(R,i1:i2,j1:j2),kron(A[ia],A[ia])) 
          copyto!(view(Y,i1:i2),C[ic][:]) 
          i1 = i2+1
          j1 = j2+1
      end
      ldiv!(qr!(I-R),Y)
      z = Vector{Matrix{T}}(undef,0)
      i1 = 1
      for i = 1:p 
          ia = mod(i-1,pa)+1
          i2 = i1+n2[ia]-1
          push!(z,reshape(view(Y,i1:i2),na[ia],na[ia]))
          i1 = i2+1
      end
      return z
   end
end  

function pdlyaps!(KSCHUR::Int, A::AbstractArray{T1,3}, C::AbstractArray{T1,3}; adj = true) where {T1<:BlasReal}
   # Standard solver for A in a periodic Schur form, with structure exploiting solution of
   # the underlying 2x2 periodic Sylvester equations. 
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
      throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   rem(pc,pa) == 0 || error("the period of C must be an integer multiple of A")
   (KSCHUR <= 0 || KSCHUR > pa ) && 
         error("KSCHUR has a value $KSCHUR, which is inconsistent with A ")

   if pa == 1 && pc == 1   
      lyapds!(view(A,:,:,1), view(C,:,:,1); adj)
      return #C[:,:,:]
   end
   ONE = one(T1)

   # allocate cache for 2x2 periodic Sylvester solver
   G = Array{T1,3}(undef,2,2,pc)
   WUSD = Array{Float64,3}(undef,4,4,pc)
   WUD = Array{Float64,3}(undef,4,4,pc)
   WUL = Matrix{Float64}(undef,4*pc,4)
   WY = Vector{Float64}(undef,4*pc)
   W = Matrix{Float64}(undef,8,8)
   qr_ws = QRWs(zeros(8), zeros(4))
   ormqr_ws = QROrmWs(zeros(4), qr_ws.τ)   

   # determine the dimensions of the diagonal blocks of real Schur form
   ba, p = MatrixEquations.sfstruct(A[:,:,KSCHUR])
   if adj
      #
      # Solve    A(j)'*X(j+1)*A(j) + C(j) = X(j) .
      #
      # The (K,L)th blocks of X(j), j = 1, ..., p are determined
      # starting from upper-left corner column by column by
      #
      #   A(j)(K,K)'*X(j+1)(K,L)*A(j)(L,L) - X(j)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # where
      #                K              L-1
      #   R(j)(K,L) = SUM {A(j)(I,K)'*SUM [X(j+1)(I,J)*A(j)(J,L)]}
      #               I=1             J=1
      #             
      #                 K-1
      #             +  {SUM [A(j)(I,K)'*X(j+1)(I,L)]}*A(j)(L,L)
      #                 I=1
      i = 1
      @inbounds  for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
              dl = ba[ll]
              j1 = j+dl-1
              l = j:j1
              Ckl = view(C,k,l,1:pc)
              y = view(G,1:dk,1:dl,1:pc)
              copyto!(y,Ckl)
              if kk > 1
                 # C(j+1)[l,k] = C(j+1)[l,ir]*A(j)[ir,k]
                 ic = 1:j1
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),view(C,l,ir,ii1),view(A,ir,k,ia))
                     #y += C(j+1)[ic,k]'*A(j)[ic,l]
                     mul!(view(y,:,:,ii),transpose(view(C,ic,k,ii1)),view(A,ic,l,ia),ONE,ONE)
                 end
              end
              dpsylv2krsol!(adj, dk, dl, KSCHUR, view(A,k,k,1:pa), view(A,l,l,1:pa), y, WUD, WUSD, WUL, WY, W, qr_ws, ormqr_ws) 
              copyto!(Ckl,y)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Ckl[1,2,ii]+Ckl[2,1,ii])
                     Ckl[1,2,ii] = temp; Ckl[2,1,ii] = temp
                 end
              end
              j += dl
              if ll < kk
                 # C(j+1)[l,k] += C(j+1)[k,l]'*A(j)[k,k]
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),transpose(view(C,k,l,ii1)),view(A,k,k,ia),ONE,ONE) 
                 end
              end
          end
          if kk > 1
             # C(j)[ir,k] = C(j)[k,ir]'
             for ii = 1:pc
                 transpose!(view(C,ir,k,ii),view(C,k,ir,ii))
             end
          end
          i += dk
      end
   else
      #
      # Solve    A(j)*X(j)*A(j)' + C(j) = X(j+1) .
      #
      # The (K,L)th block of X(j) is determined starting from
      # bottom-right corner column by column by
      #
      #    A(j)(K,K)*X(j)(K,L)*A(j)(L,L)' - X(j+1)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # Where
      #
      #                 N               N
      #    R(j)(K,L) = SUM {A(j)(K,I)* SUM [X(j)(I,J)*A(j)(L,J)']} +
      #                I=K            J=L+1
      #              
      #                N
      #             { SUM [A(j)(K,J)*X(j)(J,L)]}*A(j)(L,L)'
      #              J=K+1
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          ir = j+1:n
          for kk = p:-1:ll
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              Clk = view(C,l,k,1:pc)
              y = view(G,1:dl,1:dk,1:pc)
              copyto!(y,Clk)
              if ll < p
                 ic = i1:n
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] = C(j)[k,ir]*A(j)[l,ir]'
                     mul!(view(C,k,l,ii),view(C,k,ir,ii),transpose(view(A,l,ir,ia)))
                     # y += (A(j)[k,ic]*C(j)[ic,l])'
                     mul!(view(y,:,:,ii),transpose(view(C,ic,l,ii)),transpose(view(A,k,ic,ia)),ONE,ONE)
                 end
              end
              dpsylv2krsol!(adj, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WUD, WUSD, WUL, WY, W, qr_ws, ormqr_ws) 
              #dpsylv2!(adj, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WZ, WY)
              copyto!(Clk,y)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Clk[1,2,ii]+Clk[2,1,ii])
                     Clk[1,2,ii] = temp; Clk[2,1,ii] = temp
                 end
              end
              i -= dk
              if i >= j
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] += (A(j)[l,l]*C(j)[l,k])'
                     mul!(view(C,k,l,ii),transpose(view(C,l,k,ii)),transpose(view(A,l,l,ia)),ONE,ONE)
                 end
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             for ii = 1:pc
                 # C(j)[ir,l] = C(j)[l,ir]'
                 transpose!(view(C,ir,l,ii),view(C,l,ir,ii))
             end
          end
          j -= dl
      end
   end
   return #C[:,:,:]
end
function pdlyaps2!(KSCHUR::Int, A::AbstractArray{T1,3}, C::AbstractArray{T1,3}, E::AbstractArray{T1,3}, WORK) where {T1<:BlasReal}
   # Standard solver for A in a periodic Schur form, with structure exploiting solution of
   # the underlying 2x2 periodic Sylvester equations. 
   n = LinearAlgebra.checksquare(view(A,:,:,1))
   pa = size(A,3)
   pc = size(C,3)
   pe = size(E,3)
   LinearAlgebra.checksquare(view(C,:,:,1)) == n || throw(DimensionMismatch("C[:,:,i] must be $n x $n symmetric matrices"))
   LinearAlgebra.checksquare(view(E,:,:,1)) == n || throw(DimensionMismatch("E[:,:,i] must be $n x $n symmetric matrices"))
   # (LinearAlgebra.checksquare(view(C,:,:,1)) == n && all([issymmetric(view(C,:,:,i)) for i in 1:pc])) ||
   #    throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   rem(pc,pa) == 0 || error("the period of C must be an integer multiple of A")
   rem(pe,pa) == 0 || error("the period of E must be an integer multiple of A")
   (KSCHUR <= 0 || KSCHUR > pa ) && 
         error("KSCHUR has a value $KSCHUR, which is inconsistent with A ")
   
   if pa == 1 && pc == 1 && pe == 1  
      lyapds!(view(A,:,:,1), view(C,:,:,1); adj = false)
      lyapds!(view(A,:,:,1), view(E,:,:,1); adj = true)
      return #C[:,:,:]
   end
   ONE = one(T1)

   # use preallocated cache for 2x2 periodic Sylvester solver
   # G(2×2×pc), WUSD(4×4×pc), WUD(4×4×pc), WUL(4pc×4), WY(4pc), W(8×8), 
   # qr_ws = QRWs(zeros(8), zeros(4)), ormqr_ws = QROrmWs(zeros(4), qr_ws.τ)   
   (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws) = WORK
 
   # determine the dimensions of the diagonal blocks of real Schur form
   ba, p = MatrixEquations.sfstruct(view(A,:,:,KSCHUR))


   # Solve    A(j)*X(j)*A(j)' + C(j) = X(j+1) .
   #
   # The (K,L)th block of X(j) is determined starting from
   # bottom-right corner column by column by
   #
   #    A(j)(K,K)*X(j)(K,L)*A(j)(L,L)' - X(j+1)(K,L) = -C(j)(K,L) - R(j)(K,L)
   #
   # Where
   #
   #                 N               N
   #    R(j)(K,L) = SUM {A(j)(K,I)* SUM [X(j)(I,J)*A(j)(L,J)']} +
   #                I=K            J=L+1
   #              
   #                N
   #             { SUM [A(j)(K,J)*X(j)(J,L)]}*A(j)(L,L)'
   #              J=K+1
   j = n
   for ll = p:-1:1
      dl = ba[ll]
      l = j-dl+1:j
      i = n
      ir = j+1:n
      for kk = p:-1:ll
          dk = ba[kk]
          i1 = i-dk+1
          k = i1:i
          Clk = view(C,l,k,1:pc)
          y = view(G,1:dl,1:dk,1:pc)
          copyto!(y,Clk)
          if ll < p
             ic = i1:n
             for ii = 1:pc
                 ia = mod(ii-1,pa)+1
                 # C(j)[k,l] = C(j)[k,ir]*A(j)[l,ir]'
                 mul!(view(C,k,l,ii),view(C,k,ir,ii),transpose(view(A,l,ir,ia)))
                 # y += (A(j)[k,ic]*C(j)[ic,l])'
                 mul!(view(y,:,:,ii),transpose(view(C,ic,l,ii)),transpose(view(A,k,ic,ia)),ONE,ONE)
             end
          end
          dpsylv2krsol!(false, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WUD, WUSD, WUL, WY, W, qr_ws, ormqr_ws) 
          #dpsylv2!(adj, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WZ, WY)
          copyto!(Clk,y)
          if ll == kk && dl == 2
             for ii = 1:pc
                 temp = 0.5*(Clk[1,2,ii]+Clk[2,1,ii])
                 Clk[1,2,ii] = temp; Clk[2,1,ii] = temp
             end
          end
          i -= dk
          if i >= j
             for ii = 1:pc
                 ia = mod(ii-1,pa)+1
                 # C(j)[k,l] += (A(j)[l,l]*C(j)[l,k])'
                 mul!(view(C,k,l,ii),transpose(view(C,l,k,ii)),transpose(view(A,l,l,ia)),ONE,ONE)
             end
          else
             break
          end
      end
      if ll < p
         ir = i+2:n
         for ii = 1:pc
             # C(j)[ir,l] = C(j)[l,ir]'
             transpose!(view(C,ir,l,ii),view(C,l,ir,ii))
         end
      end
      j -= dl
   end
   # Solve    A(j)'*X(j+1)*A(j) + E(j) = X(j) .
   #
   # The (K,L)th blocks of X(j), j = 1, ..., p are determined
   # starting from upper-left corner column by column by
   #
   #   A(j)(K,K)'*X(j+1)(K,L)*A(j)(L,L) - X(j)(K,L) = -E(j)(K,L) - R(j)(K,L)
   #
   # where
   #                K              L-1
   #   R(j)(K,L) = SUM {A(j)(I,K)'*SUM [X(j+1)(I,J)*A(j)(J,L)]}
   #               I=1             J=1
   #             
   #                 K-1
   #             +  {SUM [A(j)(I,K)'*X(j+1)(I,L)]}*A(j)(L,L)
   #                 I=1
   i = 1
   @inbounds  for kk = 1:p
      dk = ba[kk]
      k = i:i+dk-1
      j = 1
      ir = 1:i-1
      for ll = 1:kk
          dl = ba[ll]
          j1 = j+dl-1
          l = j:j1
          Ekl = view(E,k,l,1:pe)
          y = view(G,1:dk,1:dl,1:pe)
          copyto!(y,Ekl)
          if kk > 1
             # E(j+1)[l,k] = E(j+1)[l,ir]*A(j)[ir,k]
             ic = 1:j1
             for ii = 1:pe
                 ia = mod(ii-1,pa)+1
                 ii1 = mod(ii,pe)+1
                 mul!(view(E,l,k,ii1),view(E,l,ir,ii1),view(A,ir,k,ia))
                 #y += E(j+1)[ic,k]'*A(j)[ic,l]
                 mul!(view(y,:,:,ii),transpose(view(E,ic,k,ii1)),view(A,ic,l,ia),ONE,ONE)
             end
          end
          dpsylv2krsol!(true, dk, dl, KSCHUR, view(A,k,k,1:pa), view(A,l,l,1:pa), y, WUD, WUSD, WUL, WY, W, qr_ws, ormqr_ws) 
           #dpsylv2!(adj, dk, dl, KSCHUR, view(A,k,k,1:pa), view(A,l,l,1:pa), y, WZ, WY)
          copyto!(Ekl,y)
          if ll == kk && dl == 2
             for ii = 1:pe
                 temp = 0.5*(Ekl[1,2,ii]+Ekl[2,1,ii])
                 Ekl[1,2,ii] = temp; Ekl[2,1,ii] = temp
             end
          end
          j += dl
          if ll < kk
             # E(j+1)[l,k] += E(j+1)[k,l]'*A(j)[k,k]
             for ii = 1:pe
                 ia = mod(ii-1,pa)+1
                 ii1 = mod(ii,pe)+1
                 mul!(view(E,l,k,ii1),transpose(view(E,k,l,ii1)),view(A,k,k,ia),ONE,ONE) 
             end
           end
      end
      if kk > 1
         # E(j)[ir,k] = E(j)[k,ir]'
         for ii = 1:pe
             transpose!(view(E,ir,k,ii),view(E,k,ir,ii))
         end
      end
      i += dk
   end
   return nothing
end

function pdlyaps3!(KSCHUR::Int, A::AbstractArray{T1,3}, C::AbstractArray{T1,3}; adj = true) where {T1<:BlasReal}
   # Alternative solver for A in a periodic Schur form, with Kronecker product expansion based solution of
   # the underlying 2x2 periodic Sylvester equations, using fine structure exploitation of small matrices.
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
      throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   rem(pc,pa) == 0 || error("the period of C must be an integer multiple of A")
   (KSCHUR <= 0 || KSCHUR > pa ) && 
         error("KSCHUR has a value $KSCHUR, which is inconsistent with A ")

   if pa == 1 && pc == 1   
      lyapds!(view(A,:,:,1), view(C,:,:,1); adj)
      return #C[:,:,:]
   end
   ONE = one(T1)

   # determine the dimensions of the diagonal blocks of real Schur form

   G = Array{T1,3}(undef,2,2,pc)
   WZ = Matrix{Float64}(undef,4*pc,max(4*pc,5))
   WY = Vector{Float64}(undef,4*pc)
   ba, p = MatrixEquations.sfstruct(A[:,:,KSCHUR])
   if adj
      #
      # Solve    A(j)'*X(j+1)*A(j) + C(j) = X(j) .
      #
      # The (K,L)th blocks of X(j), j = 1, ..., p are determined
      # starting from upper-left corner column by column by
      #
      #   A(j)(K,K)'*X(j+1)(K,L)*A(j)(L,L) - X(j)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # where
      #                K              L-1
      #   R(j)(K,L) = SUM {A(j)(I,K)'*SUM [X(j+1)(I,J)*A(j)(J,L)]}
      #               I=1             J=1
      #             
      #                 K-1
      #             +  {SUM [A(j)(I,K)'*X(j+1)(I,L)]}*A(j)(L,L)
      #                 I=1
      i = 1
      @inbounds  for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
              dl = ba[ll]
              j1 = j+dl-1
              l = j:j1
              Ckl = view(C,k,l,1:pc)
              y = view(G,1:dk,1:dl,1:pc)
              copyto!(y,Ckl)
              if kk > 1
                 # C(j+1)[l,k] = C(j+1)[l,ir]*A(j)[ir,k]
                 ic = 1:j1
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),view(C,l,ir,ii1),view(A,ir,k,ia))
                     #y += C(j+1)[ic,k]'*A(j)[ic,l]
                     mul!(view(y,:,:,ii),transpose(view(C,ic,k,ii1)),view(A,ic,l,ia),ONE,ONE)
                 end
              end
              dpsylv2!(adj, dk, dl, KSCHUR, view(A,k,k,1:pa), view(A,l,l,1:pa), y, WZ, WY)
              copyto!(Ckl,y)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Ckl[1,2,ii]+Ckl[2,1,ii])
                     Ckl[1,2,ii] = temp; Ckl[2,1,ii] = temp
                 end
              end
              j += dl
              if ll < kk
                 # C(j+1)[l,k] += C(j+1)[k,l]'*A(j)[k,k]
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),transpose(view(C,k,l,ii1)),view(A,k,k,ia),ONE,ONE) 
                 end
              end
          end
          if kk > 1
             # C(j)[ir,k] = C(j)[k,ir]'
             for ii = 1:pc
                 transpose!(view(C,ir,k,ii),view(C,k,ir,ii))
             end
          end
          i += dk
      end
   else
      #
      # Solve    A(j)*X(j)*A(j)' + C(j) = X(j+1) .
      #
      # The (K,L)th block of X(j) is determined starting from
      # bottom-right corner column by column by
      #
      #    A(j)(K,K)*X(j)(K,L)*A(j)(L,L)' - X(j+1)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # Where
      #
      #                 N               N
      #    R(j)(K,L) = SUM {A(j)(K,I)* SUM [X(j)(I,J)*A(j)(L,J)']} +
      #                I=K            J=L+1
      #              
      #                N
      #             { SUM [A(j)(K,J)*X(j)(J,L)]}*A(j)(L,L)'
      #              J=K+1
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          ir = j+1:n
          for kk = p:-1:ll
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              Clk = view(C,l,k,1:pc)
              y = view(G,1:dl,1:dk,1:pc)
              copyto!(y,Clk)
              if ll < p
                 ic = i1:n
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] = C(j)[k,ir]*A(j)[l,ir]'
                     mul!(view(C,k,l,ii),view(C,k,ir,ii),transpose(view(A,l,ir,ia)))
                     # y += (A(j)[k,ic]*C(j)[ic,l])'
                     mul!(view(y,:,:,ii),transpose(view(C,ic,l,ii)),transpose(view(A,k,ic,ia)),ONE,ONE)
                 end
              end
              dpsylv2!(adj, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WZ, WY)
              copyto!(Clk,y)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Clk[1,2,ii]+Clk[2,1,ii])
                     Clk[1,2,ii] = temp; Clk[2,1,ii] = temp
                 end
              end
              i -= dk
              if i >= j
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] += (A(j)[l,l]*C(j)[l,k])'
                     mul!(view(C,k,l,ii),transpose(view(C,l,k,ii)),transpose(view(A,l,l,ia)),ONE,ONE)
                 end
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             for ii = 1:pc
                 # C(j)[ir,l] = C(j)[l,ir]'
                 transpose!(view(C,ir,l,ii),view(C,l,ir,ii))
             end
          end
          j -= dl
      end
   end
   return #C[:,:,:]
end
function pdlyaps2!(KSCHUR::Int, A::AbstractArray{T1,3}, C::AbstractArray{T1,3}; adj = true) where {T1<:BlasReal}
   # Alternative solver for A in a periodic Schur form, with Kronecker product expansion based solution of
   # the underlying 2x2 periodic Sylvester equations. No fine structure exploitation is implemented.
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
      throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   rem(pc,pa) == 0 || error("the period of C must be an integer multiple of A")
   (KSCHUR <= 0 || KSCHUR > pa ) && 
         error("KSCHUR has a value $KSCHUR, which is inconsistent with A ")

   if pa == 1 && pc == 1   
      lyapds!(view(A,:,:,1), view(C,:,:,1); adj)
      return #C[:,:,:]
   end
   ONE = one(T1)

   # determine the dimensions of the diagonal blocks of real Schur form

   G = Array{T1,3}(undef,2,2,pc)
   WZ = Matrix{Float64}(undef,4*pc,max(4*pc,5))
   WY = Vector{Float64}(undef,4*pc)
   ba, p = MatrixEquations.sfstruct(A[:,:,KSCHUR])
   if adj
      #
      # Solve    A(j)'*X(j+1)*A(j) + C(j) = X(j) .
      #
      # The (K,L)th blocks of X(j), j = 1, ..., p are determined
      # starting from upper-left corner column by column by
      #
      #   A(j)(K,K)'*X(j+1)(K,L)*A(j)(L,L) - X(j)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # where
      #                K              L-1
      #   R(j)(K,L) = SUM {A(j)(I,K)'*SUM [X(j+1)(I,J)*A(j)(J,L)]}
      #               I=1             J=1
      #             
      #                 K-1
      #             +  {SUM [A(j)(I,K)'*X(j+1)(I,L)]}*A(j)(L,L)
      #                 I=1
      i = 1
      @inbounds  for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
              dl = ba[ll]
              j1 = j+dl-1
              l = j:j1
              Ckl = view(C,k,l,1:pc)
              y = view(G,1:dk,1:dl,1:pc)
              copyto!(y,Ckl)
              if kk > 1
                 # C(j+1)[l,k] = C(j+1)[l,ir]*A(j)[ir,k]
                 ic = 1:j1
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),view(C,l,ir,ii1),view(A,ir,k,ia))
                     #y += C(j+1)[ic,k]'*A(j)[ic,l]
                     mul!(view(y,:,:,ii),transpose(view(C,ic,k,ii1)),view(A,ic,l,ia),ONE,ONE)
                 end
              end
              _dpsylv2!(adj, dk, dl, view(A,k,k,1:pa), view(A,l,l,1:pa), y, WZ, WY)
              copyto!(Ckl,y)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Ckl[1,2,ii]+Ckl[2,1,ii])
                     Ckl[1,2,ii] = temp; Ckl[2,1,ii] = temp
                 end
              end
              j += dl
              if ll < kk
                 # C(j+1)[l,k] += C(j+1)[k,l]'*A(j)[k,k]
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),transpose(view(C,k,l,ii1)),view(A,k,k,ia),ONE,ONE) 
                 end
              end
          end
          if kk > 1
             # C(j)[ir,k] = C(j)[k,ir]'
             for ii = 1:pc
                 transpose!(view(C,ir,k,ii),view(C,k,ir,ii))
             end
          end
          i += dk
      end
   else
      #
      # Solve    A(j)*X(j)*A(j)' + C(j) = X(j+1) .
      #
      # The (K,L)th block of X(j) is determined starting from
      # bottom-right corner column by column by
      #
      #    A(j)(K,K)*X(j)(K,L)*A(j)(L,L)' - X(j+1)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # Where
      #
      #                 N               N
      #    R(j)(K,L) = SUM {A(j)(K,I)* SUM [X(j)(I,J)*A(j)(L,J)']} +
      #                I=K            J=L+1
      #              
      #                N
      #             { SUM [A(j)(K,J)*X(j)(J,L)]}*A(j)(L,L)'
      #              J=K+1
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          ir = j+1:n
          for kk = p:-1:ll
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              Clk = view(C,l,k,1:pc)
              y = view(G,1:dl,1:dk,1:pc)
              copyto!(y,Clk)
              if ll < p
                 ic = i1:n
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] = C(j)[k,ir]*A(j)[l,ir]'
                     mul!(view(C,k,l,ii),view(C,k,ir,ii),transpose(view(A,l,ir,ia)))
                     # y += (A(j)[k,ic]*C(j)[ic,l])'
                     mul!(view(y,:,:,ii),transpose(view(C,ic,l,ii)),transpose(view(A,k,ic,ia)),ONE,ONE)
                 end
              end
              _dpsylv2!(adj, dl, dk, view(A,l,l,1:pa), view(A,k,k,1:pa), y, WZ, WY)
              copyto!(Clk,y)
              i -= dk
              if i >= j
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] += (A(j)[l,l]*C(j)[l,k])'
                     mul!(view(C,k,l,ii),transpose(view(C,l,k,ii)),transpose(view(A,l,l,ia)),ONE,ONE)
                 end
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             for ii = 1:pc
                 # C(j)[ir,l] = C(j)[l,ir]'
                 transpose!(view(C,ir,l,ii),view(C,l,ir,ii))
             end
          end
          j -= dl
      end
   end
   return #C[:,:,:]
end
function pdlyaps1!(KSCHUR::Int, A::StridedArray{T1,3}, C::StridedArray{T1,3}; adj = true) where {T1<:BlasReal}
   # Alternative solver for A in a periodic Schur form, with fast iterative solution of
   # the underlying 2x2 periodic Sylvester equations. This version is usually faster 
   # than the numerically more robust implementations employing Kronecker expansion based
   # linear equations solvers.  
   n = LinearAlgebra.checksquare(A[:,:,1])
   pa = size(A,3)
   pc = size(C,3)
   (LinearAlgebra.checksquare(C[:,:,1]) == n && all([issymmetric(C[:,:,i]) for i in 1:pc])) ||
      throw(DimensionMismatch("all C[:,:,i] must be $n x $n symmetric matrices"))
   rem(pc,pa) == 0 || error("the period of C must be an integer multiple of A")
   (KSCHUR <= 0 || KSCHUR > pa ) && 
         error("KSCHUR has a value $KSCHUR, which is inconsistent with A ")

   if pa == 1 && pc == 1   
      lyapds!(view(A,:,:,1), view(C,:,:,1); adj)
      return C[:,:,:]
   end
   ONE = one(T1)

   # determine the dimensions of the diagonal blocks of real Schur form

   G = Array{T1,3}(undef,2,2,pc)
   W = Matrix{Float64}(undef,2,14)
   WX = Matrix{Float64}(undef,4,5)
   ba, p = MatrixEquations.sfstruct(A[:,:,KSCHUR])
   if adj
      #
      # Solve    A(j)'*X(j+1)*A(j) + C(j) = X(j) .
      #
      # The (K,L)th blocks of X(j), j = 1, ..., p are determined
      # starting from upper-left corner column by column by
      #
      #   A(j)(K,K)'*X(j+1)(K,L)*A(j)(L,L) - X(j)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # where
      #                K              L-1
      #   R(j)(K,L) = SUM {A(j)(I,K)'*SUM [X(j+1)(I,J)*A(j)(J,L)]}
      #               I=1             J=1
      #             
      #                 K-1
      #             +  {SUM [A(j)(I,K)'*X(j+1)(I,L)]}*A(j)(L,L)
      #                 I=1
      i = 1
      @inbounds  for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
              dl = ba[ll]
              j1 = j+dl-1
              l = j:j1
              Ckl = view(C,k,l,1:pc)
              y = view(G,1:dk,1:dl,1:pc)
              copyto!(y,Ckl)
              if kk > 1
                 # C(j+1)[l,k] = C(j+1)[l,ir]*A(j)[ir,k]
                 ic = 1:j1
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),view(C,l,ir,ii1),view(A,ir,k,ia))
                     #y += C(j+1)[ic,k]'*A(j)[ic,l]
                     mul!(view(y,:,:,ii),transpose(view(C,ic,k,ii1)),view(A,ic,l,ia),ONE,ONE)
                 end
              end
              Ckl[:,:,:] .= dpsylv2(adj, dk, dl, KSCHUR, view(A,k,k,1:pa), view(A,l,l,1:pa), y, W, WX)
              if ll == kk && dl == 2
                 for ii = 1:pc
                     temp = 0.5*(Ckl[1,2,ii]+Ckl[2,1,ii])
                     Ckl[1,2,ii] = temp; Ckl[2,1,ii] = temp
                 end
              end
              j += dl
              if ll < kk
                 # C(j+1)[l,k] += C(j+1)[k,l]'*A(j)[k,k]
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     ii1 = mod(ii,pc)+1
                     mul!(view(C,l,k,ii1),transpose(view(C,k,l,ii1)),view(A,k,k,ia),ONE,ONE) 
                 end
              end
          end
          if kk > 1
             # C(j)[ir,k] = C(j)[k,ir]'
             for ii = 1:pc
                 transpose!(view(C,ir,k,ii),view(C,k,ir,ii))
             end
          end
          i += dk
      end
   else
      #
      # Solve    A(j)*X(j)*A(j)' + C(j) = X(j+1) .
      #
      # The (K,L)th block of X(j) is determined starting from
      # bottom-right corner column by column by
      #
      #    A(j)(K,K)*X(j)(K,L)*A(j)(L,L)' - X(j+1)(K,L) = -C(j)(K,L) - R(j)(K,L)
      #
      # Where
      #
      #                 N               N
      #    R(j)(K,L) = SUM {A(j)(K,I)* SUM [X(j)(I,J)*A(j)(L,J)']} +
      #                I=K            J=L+1
      #              
      #                N
      #             { SUM [A(j)(K,J)*X(j)(J,L)]}*A(j)(L,L)'
      #              J=K+1
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          ir = j+1:n
          for kk = p:-1:ll
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              Clk = view(C,l,k,1:pc)
              y = view(G,1:dl,1:dk,1:pc)
              copyto!(y,Clk)
              if ll < p
                 ic = i1:n
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] = C(j)[k,ir]*A(j)[l,ir]'
                     mul!(view(C,k,l,ii),view(C,k,ir,ii),transpose(view(A,l,ir,ia)))
                     # y += (A(j)[k,ic]*C(j)[ic,l])'
                     mul!(view(y,:,:,ii),transpose(view(C,ic,l,ii)),transpose(view(A,k,ic,ia)),ONE,ONE)
                 end
              end
              Clk[:,:,:] .= dpsylv2(adj, dl, dk, KSCHUR, view(A,l,l,1:pa), view(A,k,k,1:pa), y, W, WX)
              i -= dk
              if i >= j
                 for ii = 1:pc
                     ia = mod(ii-1,pa)+1
                     # C(j)[k,l] += (A(j)[l,l]*C(j)[l,k])'
                     mul!(view(C,k,l,ii),transpose(view(C,l,k,ii)),transpose(view(A,l,l,ia)),ONE,ONE)
                 end
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             for ii = 1:pc
                 # C(j)[ir,l] = C(j)[l,ir]'
                 transpose!(view(C,ir,l,ii),view(C,l,ir,ii))
             end
          end
          j -= dl
      end
   end
   return C[:,:,:]
end

function dpsylv2!(adj::Bool, n1::Int, n2::Int, KSCHUR::Int, AL::StridedArray{T,3}, AR::StridedArray{T,3}, 
                  C::StridedArray{T,3}, WZ::AbstractMatrix{T}, WY::AbstractVector{T}) where {T}
#     To solve for the n1-by-n2 matrices X_j, j = 1, ..., p, 
#     1 <= n1,n2 <= 2, in the p simultaneous equations: 

#     if adj = true

#       AL_j'*X_(j+1)*AR_j - X_j = C_j, X_(p+1) = X_1  (1) 

#     or if adj = false

#       AL_j*X_j*AR_j' - X_(j+1) = C_j, X_(p+1) = X_1  (2)

#     where AL_j is n1-by-n1, AR_j is n2-by-n2, C_j is n1-by-n2.  

#     NOTE: This routine is primarily intended to be used in conjuntion 
#           with solvers for periodic Lyapunov equations. Thus, both 
#           AL and AR are formed from the diagonal blocks of the same
#           matrix in periodic real Schur form. 
#           The solution X overwrites the right-hand side C. 
#           WZ and WY are 4p-by-4p and 4p-by-1 working matrices, respectively,
#           allocated only once in the caller routine. 
#           AL and AR are assumed to have the same period, but 
#           C may have different period than AL and AR. The period
#           of C must be an integer multiple of that of AL and AR.
#           In the interests of speed, this routine does not
#                   check the inputs for errors.

#     METHOD

#     The solution is computed by explicitly forming and solving the underlying linear equation
#     Z*vec(X) = vec(C), where Z is built using Kronecker products of component matrices [1].
#     

#     REFERENCES

#     [1] A. Varga.
#         Periodic Lyapunov equations: some applications and new algorithms.
#         Int. J. Control, vol, 67, pp, 69-87, 1997.
   pa = size(AL,3)
   p = size(C,3)
   ii1 = 1:n1; ii2 = 1:n2;
   # Quick return if possible.
   if p == 1 
      MatrixEquations.lyapdsylv2!(adj, view(C, ii1, ii2, 1),  n1, n2, view(AL, ii1, ii1, 1), view(AR, ii2, ii2, 1), 
                                  view(WZ,1:4,1:4), view(WZ,1:4,5)) 
      return 
   end
   n12 = n1*n2
   N = p*n12
   R = view(WZ,1:N,1:N); copyto!(R,-I)
   #RT = zeros(T, N, N); copyto!(RT,-I)  
   Y = view(WY,1:N)
   if adj
      if n12 == 1
         ia = mod(KSCHUR+pa-1,pa)+1
         ic = mod(KSCHUR+p-1,p)+1
         R[1,N] = AR[1,1,ia]*AL[1,1,ia] 
         Y[1] = -C[1,1,ic]
         i1 = 2
         for i = p+KSCHUR-1:-1:KSCHUR+1         
             ia = mod(i-1,pa)+1
             ic = mod(i-1,p)+1
             R[i1,i1-1] = AR[1,1,ia]*AL[1,1,ia] 
             Y[i1] = -C[1,1,ic]
             i1 += 1
         end
      elseif n1 == 1 && n2 == 2
         ias = mod(KSCHUR+p-1,pa)+1
         # [ al11*ar11  al11*ar21 ]
         # [ al11*ar12  al11*ar22 ]      
         R[1,N-1] = AL[1,1,ias]*AR[1,1,ias]; R[1,N] = AL[1,1,ias]*AR[2,1,ias]
         R[2,N-1] = AL[1,1,ias]*AR[1,2,ias]; R[2,N] = AL[1,1,ias]*AR[2,2,ias]
         ic = mod(KSCHUR+p-1,p)+1
         Y[1] = -C[1,1,ic]
         Y[2] = -C[1,2,ic]
         i1 = n12+1; j1 = 1 
         for i = p+KSCHUR-1:-1:KSCHUR+1         
             ia = mod(i-1,pa)+1
             ic = mod(i-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             # [ al11*ar11          * ]
             # [ al11*ar12  al11*ar22 ]      
             R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]
             ia == ias && (R[i1,j1+1] = AL[1,1,ias]*AR[2,1,ias])
             R[i1+1,j1] = AL[1,1,ia]*AR[1,2,ia]; R[i1+1,j1+1] = AL[1,1,ia]*AR[2,2,ia]
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[1,2,ic]
             i1 = i2+1
             j1 = j2+1
         end
      elseif n1 == 2 && n2 == 1
         ias = mod(KSCHUR+p-1,pa)+1
         # [ al11*ar11  al21*ar11 ]
         # [ al12*ar11  al22*ar11 ]
         R[1,N-1] = AL[1,1,ias]*AR[1,1,ias]; R[1,N] = AL[2,1,ias]*AR[1,1,ias]
         R[2,N-1] = AL[1,2,ias]*AR[1,1,ias]; R[2,N] = AL[2,2,ias]*AR[1,1,ias]
         ic = mod(KSCHUR+p-1,p)+1
         Y[1] = -C[1,1,ic]
         Y[2] = -C[2,1,ic]
         i1 = n12+1; j1 = 1 
         for i = p+KSCHUR-1:-1:KSCHUR+1         
             ia = mod(i-1,pa)+1
             ic = mod(i-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             # [ al11*ar11          * ]
             # [ al12*ar11  al22*ar11 ]
             R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]
             ia == ias && (R[i1,j1+1] = AL[2,1,ias]*AR[1,1,ias])
             R[i1+1,j1] = AL[1,2,ia]*AR[1,1,ia]; R[i1+1,j1+1] = AL[2,2,ia]*AR[1,1,ia]
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[2,1,ic]
             i1 = i2+1
             j1 = j2+1
         end
      else
         ias = mod(KSCHUR+p-1,pa)+1
         transpose!(view(R,1:n12,N-n12+1:N),kron(view(AR,ii2,ii2,ias),view(AL,ii1,ii1,ias))) 
         ic = mod(KSCHUR+p-1,p)+1
         Y[1] = -C[1,1,ic]
         Y[2] = -C[2,1,ic]
         Y[3] = -C[1,2,ic]
         Y[4] = -C[2,2,ic]         
         i1 = n12+1; j1 = 1 
         for i = p+KSCHUR-1:-1:KSCHUR+1         
             ia = mod(i-1,pa)+1
             ic = mod(i-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             if ia == ias
                transpose!(view(R,i1:i2,j1:j2),kron(view(AR,ii2,ii2,ia),view(AL,ii1,ii1,ia))) 
             else
               # al11*ar11          0          0          0
               # al12*ar11  al22*ar11          0          0
               # al11*ar12          0  al11*ar22          0
               # al12*ar12  al22*ar12  al12*ar22  al22*ar22
                R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]
                R[i1+1,j1] = AL[1,2,ia]*AR[1,1,ia]; R[i1+1,j1+1] = AL[2,2,ia]*AR[1,1,ia]
                R[i1+2,j1] = AL[1,1,ia]*AR[1,2,ia]; R[i1+2,j1+2] = AL[1,1,ia]*AR[2,2,ia]; 
                R[i1+3,j1] = AL[1,2,ia]*AR[1,2,ia]; R[i1+3,j1+1] = AL[2,2,ia]*AR[1,2,ia]; 
                R[i1+3,j1+2] = AL[1,2,ia]*AR[2,2,ia]; R[i1+3,j1+3] = AL[2,2,ia]*AR[2,2,ia]; 
             end
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[2,1,ic]
             Y[i1+2] = -C[1,2,ic]
             Y[i1+3] = -C[2,2,ic]         
             i1 = i2+1
             j1 = j2+1
         end
      end
      ldiv!(qr!(R), Y )
      any(!isfinite, Y) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
      i1 = 1
      for i = p+KSCHUR:-1:KSCHUR+1
          ic = mod(i-1,p)+1
          i2 = i1+n12-1
          copyto!(view(C,ii1,ii2,ic), view(Y,i1:i2))  
          i1 = i2+1
      end  
   else
      if n12 == 1
         ia = mod(KSCHUR+p-1,pa)+1
         ic = mod(KSCHUR+p-1,p)+1
         R[1,N] = AR[1,1,ia]*AL[1,1,ia] 
         Y[1] = -C[1,1,ic]
         i1 = 2
         for i = 1:p-1 
             ia = mod(i+KSCHUR-1,pa)+1
             ic = mod(i+KSCHUR-1,p)+1
             R[i1,i1-1] = AR[1,1,ia]*AL[1,1,ia] 
             Y[i1] = -C[1,1,ic]
             i1 += 1
         end
      elseif n1 == 1 && n2 == 2
         ias = mod(KSCHUR+p-1,pa)+1
         # [ al11*ar11  al11*ar12 ]
         # [ al11*ar21  al11*ar22 ]      
         R[1,N-1] = AL[1,1,ias]*AR[1,1,ias]; R[1,N] = AL[1,1,ias]*AR[1,2,ias]
         R[2,N-1] = AL[1,1,ias]*AR[2,1,ias]; R[2,N] = AL[1,1,ias]*AR[2,2,ias]
         ic = mod(KSCHUR+p-1,p)+1
         Y[1] = -C[1,1,ic]
         Y[2] = -C[1,2,ic]
         i1 = n12+1; j1 = 1 
         for i = 1:p-1 
             ia = mod(i+KSCHUR-1,pa)+1
             ic = mod(i+KSCHUR-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             # [ al11*ar11  al11*ar12 ]
             # [ *          al11*ar22 ]      
             R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]; R[i1,j1+1] = AL[1,1,ia]*AR[1,2,ia]
             ia == ias && (R[i1+1,j1] = AL[1,1,ia]*AR[2,1,ia])
             R[i1+1,j1+1] = AL[1,1,ia]*AR[2,2,ia]
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[1,2,ic]
             i1 = i2+1
             j1 = j2+1
         end
      elseif n1 == 2 && n2 == 1
         ias = mod(KSCHUR+p-1,pa)+1
         # [ al11*ar11  al12*ar11 ]
         # [ al21*ar21  al22*ar11 ]      
         R[1,N-1] = AL[1,1,ias]*AR[1,1,ias]; R[1,N] = AL[1,2,ias]*AR[1,1,ias]
         R[2,N-1] = AL[2,1,ias]*AR[1,1,ias]; R[2,N] = AL[2,2,ias]*AR[1,1,ias]
         ic = mod(KSCHUR+p-1,p)+1
         Y[1] = -C[1,1,ic]
         Y[2] = -C[2,1,ic]
         i1 = n12+1; j1 = 1 
         for i = 1:p-1 
             ia = mod(i+KSCHUR-1,pa)+1
             ic = mod(i+KSCHUR-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             # [ al11*ar11  al12*ar11 ]
             # [ *          al22*ar11 ]      
             R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]; R[i1,j1+1] = AL[1,2,ia]*AR[1,1,ia]
             ia == ias && (R[i1+1,j1] = AL[2,1,ia]*AR[1,1,ia])
             R[i1+1,j1+1] = AL[2,2,ia]*AR[1,1,ia]
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[2,1,ic]
             i1 = i2+1
             j1 = j2+1
         end
      else
         ias = mod(KSCHUR+p-1,pa)+1
         ic = mod(KSCHUR+p-1,p)+1
         copyto!(view(R,1:n12,N-n12+1:N),kron(view(AR,ii2,ii2,ias),view(AL,ii1,ii1,ias))) 
         copyto!(view(Y,1:n12), view(C,ii1,ii2,ic)) 
         Y[1] = -C[1,1,ic]
         Y[2] = -C[2,1,ic]
         Y[3] = -C[1,2,ic]
         Y[4] = -C[2,2,ic]         
         i1 = n12+1; j1 = 1 
         for i = 1:p-1 
             ia = mod(i+KSCHUR-1,pa)+1
             ic = mod(i+KSCHUR-1,p)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             #R[i1:i2,j1:j2] = kron(view(AR,ii2,ii2,ia),view(AL,ii1,ii1,ia))
             if ia == ias
                copyto!(view(R,i1:i2,j1:j2),kron(view(AR,ii2,ii2,ia),view(AL,ii1,ii1,ia))) 
             else
                # al11*ar11  al12*ar11  al11*ar12  al12*ar12
                # 0          al22*ar11          0  al22*ar12
                # 0          0          al11*ar22  al12*ar22
                # 0          0          0          al22*ar22
                R[i1,j1] = AL[1,1,ia]*AR[1,1,ia]; R[i1,j1+1] = AL[1,2,ia]*AR[1,1,ia];
                R[i1,j1+2] = AL[1,1,ia]*AR[1,2,ia]; R[i1,j1+3] = AL[1,2,ia]*AR[1,2,ia];
                R[i1+1,j1+1] = AL[2,2,ia]*AR[1,1,ia]; R[i1+1,j1+3] = AL[2,2,ia]*AR[1,2,ia]
                R[i1+2,j1+2] = AL[1,1,ia]*AR[2,2,ia]; R[i1+2,j1+3] = AL[1,2,ia]*AR[2,2,ia]; 
                R[i1+3,j1+3] = AL[2,2,ia]*AR[2,2,ia]; 
             end
             Y[i1] = -C[1,1,ic]
             Y[i1+1] = -C[2,1,ic]
             Y[i1+2] = -C[1,2,ic]
             Y[i1+3] = -C[2,2,ic]         
             i1 = i2+1
             j1 = j2+1
         end
      end
      ldiv!(qr!(R), Y )
      any(!isfinite, Y) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
      i1 = 1
      for i = 1:p
          ic = mod(i+KSCHUR-1,p)+1
          i2 = i1+n12-1
          copyto!(view(C,ii1,ii2,ic), view(Y,i1:i2))  
          i1 = i2+1
      end  
   end
end

function kronset!(R::AbstractMatrix{T}, adj::Bool, n1::Int, n2::Int, SCHUR::Bool, AL::StridedMatrix{T}, AR::StridedMatrix{T}) where {T}
   n12 = n1*n2
   if n12 == 1
      R[1,1] = AR[1,1]*AL[1,1]
      return
   end
   if adj
      if n1 == 1 && n2 == 2
         # [ al11*ar11  al11*ar21 ]
         # [ al11*ar12  al11*ar22 ]      
         R[1,1] = AL[1,1]*AR[1,1]; 
         R[1,2] = SCHUR ? AL[1,1]*AR[2,1] : zero(T)
         R[2,1] = AL[1,1]*AR[1,2]; R[2,2] = AL[1,1]*AR[2,2]
      elseif n1 == 2 && n2 == 1
         # [ al11*ar11  al21*ar11 ]
         # [ al12*ar11  al22*ar11 ]
         R[1,1] = AL[1,1]*AR[1,1]; 
         R[1,2] = SCHUR ? AL[2,1]*AR[1,1] : zero(T)
         R[2,1] = AL[1,2]*AR[1,1]; R[2,2] = AL[2,2]*AR[1,1]
      else
         i12 = 1:n12
         if SCHUR
            ii1 = 1:n1; ii2 = 1:n2
            kron!(view(R,i12,i12),view(AR,ii2,ii2),view(AL,ii1,ii1)) 
            _transpose!(view(R,i12,i12))  
         else
               # al11*ar11          0          0          0
               # al12*ar11  al22*ar11          0          0
               # al11*ar12          0  al11*ar22          0
               # al12*ar12  al22*ar12  al12*ar22  al22*ar22
            R[1,1] = AL[1,1]*AR[1,1]
            R[2,1] = AL[1,2]*AR[1,1]; R[2,2] = AL[2,2]*AR[1,1]
            R[3,1] = AL[1,1]*AR[1,2]; R[3,2] = zero(T); R[3,3] = AL[1,1]*AR[2,2]; 
            R[4,1] = AL[1,2]*AR[1,2]; R[4,2] = AL[2,2]*AR[1,2]; 
            R[4,3] = AL[1,2]*AR[2,2]; R[4,4] = AL[2,2]*AR[2,2]; 
            tril!(view(R,i12,i12))
         end
      end      
   else
      if n1 == 1 && n2 == 2
         # [ al11*ar11  al11*ar12 ]
         # [ al11*ar21  al11*ar22 ]      
         R[1,1] = AL[1,1]*AR[1,1]; R[1,2] = AL[1,1]*AR[1,2]
         R[2,1] = SCHUR ? AL[1,1]*AR[2,1] : zero(T)
         R[2,2] = AL[1,1]*AR[2,2]
      elseif n1 == 2 && n2 == 1
         # [ al11*ar11  al12*ar11 ]
         # [ al21*ar21  al22*ar11 ]      
         R[1,1] = AL[1,1]*AR[1,1]; R[1,2] = AL[1,2]*AR[1,1]
         R[2,1] = SCHUR ? AL[2,1]*AR[1,1]  : zero(T)
         R[2,2] = AL[2,2]*AR[1,1]
      else
         i12 = 1:n12
         if SCHUR
            ii1 = 1:n1; ii2 = 1:n2
            kron!(view(R,i12,i12),view(AR,ii2,ii2),view(AL,ii1,ii1))
         else
            # al11*ar11  al12*ar11  al11*ar12  al12*ar12
            # 0          al22*ar11          0  al22*ar12
            # 0          0          al11*ar22  al12*ar22
            # 0          0          0          al22*ar22
            R[1,1] = AL[1,1]*AR[1,1]; R[1,2] = AL[1,2]*AR[1,1];
            R[1,3] = AL[1,1]*AR[1,2]; R[1,4] = AL[1,2]*AR[1,2];
            R[2,2] = AL[2,2]*AR[1,1]; R[2,3] = zero(T); R[2,4] = AL[2,2]*AR[1,2]
            R[3,3] = AL[1,1]*AR[2,2]; R[3,4] = AL[1,2]*AR[2,2]; 
            R[4,4] = AL[2,2]*AR[2,2]; 
            triu!(view(R,i12,i12))
         end
      end
   end
   return nothing
end
function _transpose!(A::AbstractMatrix)
   n = LinearAlgebra.checksquare(A)
   for j = 1:n
       for i = 1:j-1
           temp = A[i,j]
           A[i,j] = A[j,i]
           A[j,i] = temp
       end
   end
end   

function dpsylv2krsol!(adj::Bool, n1::Int, n2::Int, KSCHUR::Int, AL::StridedArray{T,3}, AR::StridedArray{T,3}, 
                  C::StridedArray{T,3}, WUD::AbstractArray{T,3}, WUSD::AbstractArray{T,3}, WUL::AbstractMatrix{T},  WY::AbstractVector{T}, W::AbstractMatrix{T}, qr_ws, ormqr_ws) where {T}
#     To solve for the n1-by-n2 matrices X_j, j = 1, ..., p, 
#     1 <= n1,n2 <= 2, in the p simultaneous equations: 

#     if adj = true

#       AL_j'*X_(j+1)*AR_j - X_j = C_j, X_(p+1) = X_1  (1) 

#     or if adj = false

#       AL_j*X_j*AR_j' - X_(j+1) = C_j, X_(p+1) = X_1  (2)

#     where AL_j is n1 by n1, AR_j is n2 by n2, C_j is n1 by n2.  

#     NOTE: This routine is primarily intended to be used in conjuntion 
#           with solvers for periodic Lyapunov equations. Thus, both 
#           AL and AR are formed from the diagonal blocks of the same
#           matrix in periodic real Schur form. 
#           The solution X overwrites the right-hand side C. 
#           WUD and WUSD are 4x4xp 3-dimensional arrays, WUL and W are 4px4 and 8x4 matrices,
#           WY is a 4p-dimensional vector. 
#           AL and AR are assumed to have the same period, but 
#           C may have different period than AL and AR. The period
#           of C must be an integer multiple of that of AL and AR.
#           In the interests of speed, this routine does not
#                   check the inputs for errors.

#     METHOD

#     The solution is computed by explicitly forming and solving the underlying linear equation
#     Z*vec(X) = vec(C), using Kronecker products of component matrices [1]. 
#     Only the diagonal, supra-diagonal and last column blocks of Z are explicitly built.  
#     A structure exploiting QR-factorization based solution method is employed, using 
#     Algorithm 3 of [1], with the LU factorizations replaced by QR-factorizations.  
#     
#     REFERENCES
#     [1] A. Varga.
#         Periodic Lyapunov equations: some applications and new algorithms.
#         Int. J. Control, vol, 67, pp, 69-87, 1997.
   pa = size(AL,3)
   p = size(C,3)
   ii1 = 1:n1; ii2 = 1:n2;
   # Quick return if possible.
   if p == 1 
      MatrixEquations.lyapdsylv2!(adj, view(C, ii1, ii2, 1),  n1, n2, view(AL, ii1, ii1, 1), view(AR, ii2, ii2, 1), 
                                  view(WUL,1:4,1:4), view(WY,1:4)) 
      return 
   end
   n12 = n1*n2
   n22 = 2*n12
   N = p*n12
   i1 = 1:n12; i2 = n12+1:n22
   USD = view(WUSD,i1,i1,1:p-1)
   UD = view(WUD,i1,i1,1:p)
   UK = view(WUL,1:N,i1)
   YK = view(WY,1:N)
   zmi = view(W,1:n22,i1)
   uu = view(W,1:n22,i2)
   length(qr_ws.τ) == n12 || resize!(qr_ws.τ,n12)

   
   ias = mod(KSCHUR+p-1,pa)+1
   ic = mod(KSCHUR+p-1,p)+1
   copyto!(view(YK,i1),view(C,ii1,ii2,ic))
   kronset!(view(UK,i1,i1), adj, n1, n2, true, view(AL,ii1,ii1,ias), view(AR,ii2,ii2,ias)) 
   fill!(view(UK,n12+1:N-n12,i1),zero(T))
   copyto!(view(UK,N-n12+1:N,i1),-I)
   copyto!(view(YK,i1),view(C,ii1,ii2,ic))


   # Build the blocks of the bordered almost block diagonal (BABD) system and the right-hand side 
   if adj
      j = 1; j1 = 1; 
      for i = p+KSCHUR-1:-1:KSCHUR+1   
          j1 += n12      
          ia = mod(i-1,pa)+1
          ic = mod(i-1,p)+1
          kronset!(view(USD,:,:,j), adj, n1, n2, ia == ias, view(AL,ii1,ii1,ia), view(AR,ii2,ii2,ia)) 
          copyto!(view(YK,j1:j1+n12-1),view(C,ii1,ii2,ic))
          j += 1
      end
   else
      j = 1; j1 = 1; 
      for i = 1:p-1 
          j1 += n12      
          ia = mod(i+KSCHUR-1,pa)+1
          ic = mod(i+KSCHUR-1,p)+1
          kronset!(view(USD,:,:,i), adj, n1, n2, ia == ias, view(AL,ii1,ii1,ia), view(AR,ii2,ii2,ia)) 
          copyto!(view(YK,j1:j1+n12-1),view(C,ii1,ii2,ic))
      end
   end
   for i = 1:N
      YK[i] = -YK[i]
   end

   # Solve the BABD system H*y = g using Algorithm 3 of [1] with LU factorizations replaced by QR decompositions 
   # First compute the QR factorization H = Q*R and update g <- Q'*g
   j1 = 1; 
   il = N-n12+1:N
   for j = 1:p-1          
       if j == 1
          copyto!(view(uu,i1,i1), -I)
       else
          copyto!(view(uu,i1,i1),view(UD,:,:,j))      
       end
       copyto!(view(uu,i2,i1),view(USD,:,:,j))      
       #F = qr!(uu)
       FastLapackInterface.LAPACK.geqrf!(qr_ws, uu; resize = false)
       #copy!(view(UD,:,:,j), F.R)
       set_R!(view(UD,:,:,j), uu)
       # lmul!(F.Q',view(UK,j1:j1+n22-1,i1))
       LAPACK.ormqr!(ormqr_ws, 'L', 'T', uu, view(UK,j1:j1+n22-1,i1))
       #lmul!(F.Q',view(YK,j1:j1+n22-1))
       LAPACK.ormqr!(ormqr_ws, 'L', 'T', uu, view(YK,j1:j1+n22-1))
       fill!(view(zmi,i1,i1), zero(T))
       copyto!(view(zmi,i2,i1),-I)    
       #lmul!(F.Q',zmi)
       LAPACK.ormqr!(ormqr_ws, 'L', 'T', uu, zmi)
       copyto!(view(USD,:,:,j), view(zmi,i1,:))
       copyto!(view(UD,:,:,j+1), view(zmi,i2,:))
       j1 += n12
   end
   # F = qr!(view(UK,il,i1))
   # copyto!(view(UD,:,:,p), F.R)
   # lmul!(F.Q',view(YK,il,1:1))
   LAPACK.geqrf!(qr_ws, view(UK,il,i1))
   set_R!(view(UD,:,:,p), view(UK,il,i1))
   LAPACK.ormqr!(ormqr_ws, 'L', 'T',view(UK,il,i1), view(YK,il))


   # Solve R*y = g by overwritting g
   ldiv!(UpperTriangular(view(UD,:,:,p)),view(YK,il))
   il1 = il .- n12
   mul!(view(YK,il1),view(UK,il1,i1),view(YK,il),-1,1)
   ldiv!(UpperTriangular(view(UD,:,:,p-1)),view(YK,il1))
   for i = p-2:-1:1
       il2 = il1 .- n12
       mul!(view(YK,il2),view(UK,il2,i1),view(YK,il),-1,1)
       mul!(view(YK,il2),view(USD,:,:,i),view(YK,il1),-1,1)
       ldiv!(UpperTriangular(view(UD,:,:,i)),view(YK,il2)) 
       il1 = il1 .- n12
   end

   any(!isfinite, YK) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
 
   # Reorder solution blocks
   if adj
      i1 = 1
      for i = p+KSCHUR:-1:KSCHUR+1
          ic = mod(i-1,p)+1
          i2 = i1+n12-1
          copyto!(view(C,ii1,ii2,ic), view(YK,i1:i2))  
          i1 = i2+1
      end
   else
      i1 = 1
      for i = 1:p
          ic = mod(i+KSCHUR-1,p)+1
          i2 = i1+n12-1
          copyto!(view(C,ii1,ii2,ic), view(YK,i1:i2))  
          i1 = i2+1
      end  
   end
   return nothing
end
function set_R!(R::AbstractMatrix, U::AbstractMatrix)
   m, n = size(U)
   if m >= n
      p = LinearAlgebra.checksquare(R)
      @assert p == n
      pm = n
   else
      p = m
      @assert (m, n) == size(R)
      pm = n
   end
   ZERO = zero(eltype(R))
   for j in 1:p
       R[j,j] = U[j,j]      
       for i in 1:j-1
           R[i,j] = U[i,j]
           R[j,i] = ZERO
       end
   end
   pm > p && copyto!(view(R,:,p+1:n),view(U,:,p+1:n))
   return nothing
end

function _dpsylv2!(adj::Bool, n1::Int, n2::Int, AL::StridedArray{T,3}, AR::StridedArray{T,3}, 
                  C::StridedArray{T,3}, WZ::AbstractMatrix{T}, WY::AbstractVector{T}) where {T}
#     To solve for the n1-by-n2 matrices X_j, j = 1, ..., p, 
#     1 <= n1,n2 <= 2, in the p simultaneous equations: 

#     if adj = true

#       AL_j'*X_(j+1)*AR_j - X_j = C_j, X_(p+1) = X_1  (1) 

#     or if adj = false

#       AL_j*X_j*AR_j' - X_(j+1) = C_j, X_(p+1) = X_1  (2)

#     where AL_j is n1 by n1, AR_j is n2 by n2, C_j is n1 by n2.  

#     NOTE: This routine is primarily intended to be used in conjuntion 
#           with solvers for periodic Lyapunov equations. Thus, both 
#           AL and AR are formed from the diagonal blocks of the same
#           matrix in periodic real Schur form. 
#           The solution X overwrites the right-hand side C. 
#           WZ and WY are 4p-by-4p and 4p-by-1 working matrices, respectively,
#           allocated only once in the caller routine. 
#           AL and AR are assumed to have the same period, but 
#           C may have different period than AL and AR. The period
#           of C must be an integer multiple of that of AL and AR.
#           In the interests of speed, this routine does not
#                   check the inputs for errors.

#     METHOD

#     The solution is computed by explicitly forming and solving the underlying linear equation
#     Z*vec(X) = vec(C), where Z is built using Kronecker products of component matrices [1].
#     

#     REFERENCES

#     [1] A. Varga.
#         Periodic Lyapunov equations: some applications and new algorithms.
#         Int. J. Control, vol, 67, pp, 69-87, 1997.
   pa = size(AL,3)
   p = size(C,3)
   ii1 = 1:n1; ii2 = 1:n2;
   # Quick return if possible.
   if p == 1 
      MatrixEquations.lyapdsylv2!(adj, view(C, ii1, ii2, 1),  n1, n2, view(AL, ii1, ii1, 1), view(AR, ii2, ii2, 1), 
                                  view(WZ,1:4,1:4), view(WZ,1:4,5)) 
      return 
   end
   n12 = n1*n2
   N = p*n12
   R = view(WZ,1:N,1:N)
   R = zeros(T, N, N)
   Y = view(WY,1:N)
   if adj
      if n12 == 1
         R[1,N] = AR[1,1,pa]*AL[1,1,pa] 
         R[1,1] = -1
         i1 = 2
         for i = p-1:-1:1         
             ia = mod(i-1,pa)+1
             R[i1,i1-1] = AR[1,1,ia]*AL[1,1,ia] 
             R[i1,i1] = -1
             i1 += 1
         end
      else
         R[1:n12,N-n12+1:N] = transpose(kron(view(AR,ii2,ii2,pa),view(AL,ii1,ii1,pa))) 
         copyto!(view(R,1:n12,1:n12),-I)
         i1 = n12+1; j1 = 1 
         for i = p-1:-1:1         
             ia = mod(i-1,pa)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             R[i1:i2,j1:j2] = transpose(kron(view(AR,ii2,ii2,ia),view(AL,ii1,ii1,ia)))
             copyto!(view(R,i1:i2,i1:i2),-I)
             i1 = i2+1
             j1 = j2+1
         end
         end
      reverse!(C,dims = 3)
      copyto!(Y, view(C,ii1,ii2,1:p))
      ldiv!(qr!(R), Y )
      any(!isfinite, Y) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
      copyto!(view(C,ii1,ii2,1:p), lmul!(-1,Y))
      reverse!(C,dims = 3)
   else
      if n12 == 1
         R[1,N] = AR[1,1,pa]*AL[1,1,pa] 
         R[1,1] = -1
         Y[1] = C[1,1,p]
         i1 = 2
         for i = 1:p-1         
             ia = mod(i-1,pa)+1
             R[i1,i1-1] = AR[1,1,ia]*AL[1,1,ia] 
             R[i1,i1] = -1
             Y[i1] = C[1,1,i]
             i1 += 1
         end
      else
         R[1:n12,N-n12+1:N] = kron(view(AR,ii2,ii2,pa),view(AL,ii1,ii1,pa)) 
         copyto!(view(R,1:n12,1:n12),-I)
         copyto!(view(Y,1:n12), view(C,ii1,ii2,p)) 
         i1 = n12+1; j1 = 1 
         for i = 1:p-1
             ia = mod(i-1,pa)+1
             i2 = i1+n12-1
             j2 = j1+n12-1
             R[i1:i2,j1:j2] = kron(view(AR,ii2,ii2,ia),view(AL,ii1,ii1,ia))
             copyto!(view(R,i1:i2,i1:i2),-I)
             copyto!(view(Y,i1:i2), view(C,ii1,ii2,i)) 
             i1 = i2+1
             j1 = j2+1
         end
      end
      ldiv!(qr!(R), Y )
      any(!isfinite, Y) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
      copyto!(view(C,ii1,ii2,1:p), lmul!(-1,Y))
   end
end

function dpsylv2(REV::Bool, N1::Int, N2::Int, KSCHUR::Int, TL::StridedArray{T,3}, TR::StridedArray{T,3}, 
                 B::StridedArray{T,3}, W::AbstractMatrix{T}, WX::AbstractMatrix{T}) where {T}
#     To solve for the N1-by-N2 matrices X_j, j = 1, ..., P, 
#     1 <= N1,N2 <= 2, in the P simultaneous equations: 

#     if REV = true

#       TL_j'*X_(j+1)*TR_j - X_j = B_j, X_(P+1) = X_1  (1) 

#     or if REV = false

#       TL_j*X_j*TR_j' - X_(j+1) = B_j, X_(P+1) = X_1  (2)

#     where TL_j is N1 by N1, TR_j is N2 by N2, B_j is N1 by N2.  

#     NOTE: This routine is primarily intended to be used in conjuntion 
#           with solvers for periodic Lyapunov equations. Thus, both 
#           TL and TR are formed from the diagonal blocks of the same
#           matrix in periodic real Schur form. W and WX are working matrices
#           allocated only once in the caller routine. 
#           TL and TR are assumed to have the same period, but 
#           B may have different period than TL and TR. The period
#           of B must be an integer multiple of that of TL and TR.
#           In the interests of speed, this routine does not
#                   check the inputs for errors.

#     METHOD

#     An initial approximation X_j, j=1, ..., P, is computed by reducing
#     the system (1) or (2) to an equivalent single Lyapunov equation. 
#     Then, the accuracy of the solution is iteratively improved by 
#     performing forward or backward sweeps depending on the stability
#     of eigenvalues. A maximum of 30 sweeps are performed. 

#     REFERENCES

#     [1] A. Varga.
#         Periodic Lyapunov equations: some applications and new algorithms.
#         Int. J. Control, vol, 67, pp, 69-87, 1997.

	IND(J,P) = mod( J-1, P ) + 1
   P = size(TL,3)
   PB = size(B,3)
   i1 = 1:N1; i2 = 1:N2;
   Xw = view(WX,:,1:4)
   Yw = view(WX,:,5)
   Z = view(W,:,7:8)
   X = Array{T,3}(undef, N1, N2, PB)

   # Quick return if possible.
   (N1 == 0 || N2 == 0) && (return X)
   if P == 1 && PB == 1
      copyto!(view(X,i1,i2,1), view(B,i1,i2,1))
      MatrixEquations.lyapdsylv2!(REV, view(X, i1, i2, 1),  N1, N2, view(TL, i1, i1, 1), view(TR, i2, i2, 1), Xw, Yw) 
      return X[:,:,:]
   end
   # partition working space
   AL = view(W,:,1:2)
   AR = view(W,:,3:4)
   Q = view(W,:,5:6)
   ZOLD = view(W,:,9:10)
   ATMP = view(W,:,11:12)
   BTMP = view(W,:,13:14)
   
   EPSM = 2*eps(T)

#     Define

#       AL(i,j) := TL_{j+i-1}*...*TL_{j+1}*TL_{j}, AL(j,j) := I;
#       AR(i,j) := TR_{j+i-1}*...*TR_{j+1}*TR_{j}, AR(j,j) := I;

#     If REV = true, compute an initial approximation for 
#     X_{KSCHUR+1} = Z by solving 
  
#       AL(KSCHUR+P+1,KSCHUR+1)'*Z*AR(KSCHUR+P+1,KSCHUR+1)  - Z = Qr(KSCHUR+1) 

#     where 

#       Qr(j)  = B_{j} + TL_{j}'*Qr(j+1)*TR_{j},  Qr(KSCHUR) = B(KSCHUR), 
#                for j = KSCHUR+P-1, ..., KSCHUR+1

#     If REV = false, compute an initial approximation for 
#     X_{KSCHUR} = Z by solving 
  
#       AL(KSCHUR+P,KSCHUR)*Z*AR(KSCHUR+P,KSCHUR)'  - Z = Qf(KSCHUR) 

#     where 

#       Qf(j)  = B_{j} + TL_{j}*Qf(j-1)*TR_{j}',  Qf(KSCHUR) = B(KSCHUR), 
#                for j = KSCHUR+1, ..., KSCHUR+P-1
   x = 1
   if REV 
      L1 = KSCHUR+PB-1
      L2 = KSCHUR+1
      LSTEP = -1
   else
      L1 = KSCHUR+1
      L2 = KSCHUR+PB-1
      LSTEP = 1
   end
   AL[ i1, i1 ] = TL[ i1, i1, KSCHUR ]
   AR[ i2, i2 ] = TR[ i2, i2, KSCHUR ]
   Q[i1,i2] = B[i1,i2,KSCHUR]

   if  N1 == 1 && N2 == 1 
       for JJ = L1:LSTEP:L2
           J = IND( JJ, P )
           JB = IND( JJ, PB )
           X11 =  TL[ 1, 1, J ]
           Y11 = TR[ 1, 1, J ]
           AL[ 1, 1 ] = AL[ 1, 1] * X11
           AR[ 1, 1 ] = AR[ 1, 1] * Y11
           Q[ 1, 1 ] = B[ 1, 1, JB] + X11 * Q[ 1, 1 ] * Y11
       end
   elseif N1 == 1 && N2 == 2 
       for JJ = L1:LSTEP:L2
           J = IND( JJ, P )
           JB = IND( JJ, PB )
           X11 =  TL[ 1, 1, J ]
           AL[ 1, 1 ] = AL[ 1, 1] * X11
           Q[1,1] = X11*Q[1,1]
           Q[1,2] = X11*Q[1,2]

           Y11 = TR[ 1, 1, J ]
           Y22 = TR[ 2, 2, J ]
           Y12 = TR[ 1, 2, J ]
           if REV 
               AR[1,2] = AR[1,1]*Y12 + AR[1,2]*Y22
               AR[1,1] = AR[1,1]*Y11 
               AR[2,2] = AR[2,1]*Y12 + AR[2,2]*Y22
               AR[2,1] = AR[2,1]*Y11
               Q[1,2] = Q[1,1]*Y12 + Q[1,2]*Y22 + B[ 1, 2, JB ]
               Q[1,1] = Q[1,1]*Y11 + B[ 1, 1, JB ]
           else
               AR[1,1] = Y11*AR[1,1] + Y12*AR[2,1]
               AR[2,1] = Y22*AR[2,1]
               AR[1,2] = Y11*AR[1,2] + Y12*AR[2,2]
               AR[2,2] = Y22*AR[2,2]
               Q[1,1] = Q[1,1]*Y11 + Q[1,2]*Y12 + B[ 1, 1, JB ]
               Q[1,2] = Q[1,2]*Y22 + B[ 1, 2, JB ]
           end
       end
   elseif N1 == 2 && N2 == 1 
       for JJ = L1:LSTEP:L2
           J = IND( JJ, P )
           JB = IND( JJ, PB )
           X11 = TL[ 1, 1, J ]
           X12 = TL[ 1, 2, J ]
           X22 = TL[ 2, 2, J ]
           if REV 
              AL[1,2] = AL[1,1]*X12 + AL[1,2]*X22
              AL[1,1] = AL[1,1]*X11 
              AL[2,2] = AL[2,1]*X12 + AL[2,2]*X22
              AL[2,1] = AL[2,1]*X11
              Q[2,1] = X12*Q[1,1] + X22*Q[2,1]
              Q[1,1] = X11*Q[1,1]
           else
              AL[1,1] = X11*AL[1,1] + X12*AL[2,1]
              AL[2,1] = X22*AL[2,1]
              AL[1,2] = X11*AL[1,2] + X12*AL[2,2]
              AL[2,2] = X22*AL[2,2]
              Q[1,1] = X11*Q[1,1] + X12*Q[2,1]
              Q[2,1] = X22*Q[2,1]
            end

            Y11 = TR[ 1, 1, J ]
            AR[1,1] = AR[1,1]*Y11
            Q[1,1] = Q[1,1]*Y11 + B[ 1, 1, JB ]
            Q[2,1] = Q[2,1]*Y11 + B[ 2, 1, JB ]
       end
   elseif N1 == 2 && N2 == 2 
       for JJ = L1:LSTEP:L2
          J = IND( JJ, P )
          JB = IND( JJ, PB )
          X11 = TL[ 1, 1, J ]
          X12 = TL[ 1, 2, J ]
          X22 = TL[ 2, 2, J ]
          if REV 
             AL[1,2] = AL[1,1]*X12 + AL[1,2]*X22
             AL[1,1] = AL[1,1]*X11 
             AL[2,2] = AL[2,1]*X12 + AL[2,2]*X22
             AL[2,1] = AL[2,1]*X11
             Q[2,1] = X12*Q[1,1] + X22*Q[2,1]
             Q[1,1] = X11*Q[1,1]
             Q[2,2] = X12*Q[1,2] + X22*Q[2,2]
             Q[1,2] = X11*Q[1,2]
          else
             AL[1,1] = X11*AL[1,1] + X12*AL[2,1]
             AL[2,1] = X22*AL[2,1]
             AL[1,2] = X11*AL[1,2] + X12*AL[2,2]
             AL[2,2] = X22*AL[2,2]
             Q[1,1] = X11*Q[1,1] + X12*Q[2,1]
             Q[2,1] = X22*Q[2,1]
             Q[1,2] = X11*Q[1,2] + X12*Q[2,2]
             Q[2,2] = X22*Q[2,2]
          end

          Y11 = TR[ 1, 1, J ]
          Y12 = TR[ 1, 2, J ]
          Y22 = TR[ 2, 2, J ]
          if REV 
             AR[1,2] = AR[1,1]*Y12 + AR[1,2]*Y22
             AR[1,1] = AR[1,1]*Y11 
             AR[2,2] = AR[2,1]*Y12 + AR[2,2]*Y22
             AR[2,1] = AR[2,1]*Y11
             Q[1,2] = Q[1,1]*Y12 + Q[1,2]*Y22 + B[ 1, 2, JB ]
             Q[1,1] = Q[1,1]*Y11 + B[ 1, 1, JB ]
             Q[2,2] = Q[2,1]*Y12 + Q[2,2]*Y22 + B[ 2, 2, JB ]
             Q[2,1] = Q[2,1]*Y11 + B[ 2, 1, JB ]
          else
             AR[1,1] = Y11*AR[1,1] + Y12*AR[2,1]
             AR[2,1] = Y22*AR[2,1]
             AR[1,2] = Y11*AR[1,2] + Y12*AR[2,2]
             AR[2,2] = Y22*AR[2,2]
             Q[1,1] = Q[1,1]*Y11 + Q[1,2]*Y12 + B[ 1, 1, JB ]
             Q[1,2] = Q[1,2]*Y22 + B[ 1, 2, JB ]
             Q[2,1] = Q[2,1]*Y11 + Q[2,2]*Y12 + B[ 2, 1, JB ]
             Q[2,2] = Q[2,2]*Y22 + B[ 2, 2, JB ]
          end
       end
   end
   Z[i1,i2] = -Q[i1,i2]
#     Compute X_[KSCHUR+1] (if REV=.TRUE.) or X_KSCHUR (if REV=.FALSE.).
   MatrixEquations.lyapdsylv2!(REV, view(Z,i1,i2), N1, N2, view(AL,i1,i1), view(AR,i2,i2), Xw, Yw) 
   XNORM = norm(view(Z,i1,i2), Inf)
   XNORM == 0 && (XNORM = one(T))
  
#     Determine the type of iteration to use

   if N1 == 1
      EL = abs( AL[1,1] )
   else
      EL = sqrt( abs( AL[1,1]*AL[2,2] - AL[1,2]*AL[2,1] ) )
   end
   if N2 == 1
      ER = abs( AR[1,1] )
   else
      ER = sqrt( abs( AR[1,1]*AR[2,2] - AR[1,2]*AR[2,1] ) )
   end
   STAB = EL*ER < one(T)

#     Save initial X_[KSCHUR+1].

   copyto!(view(ZOLD,i1,i2), view(Z,i1,i2))

#     Set iteration indices.

   if REV 
      if STAB 
          L1 = KSCHUR + PB
          L2 = KSCHUR + 1
          LSTEP = -1
          ITIND = -1
      else
          L1 = KSCHUR + 1
          L2 = KSCHUR + PB 
          LSTEP = 1
          ITIND = 0
      end
   else
      if STAB 
          L1 = KSCHUR 
          L2 = KSCHUR + PB - 1 
          LSTEP = 1
          ITIND = 0
      else
          L1 = KSCHUR + PB - 1 
          L2 = KSCHUR 
          LSTEP = -1
          ITIND = -1
      end
   end
   for ITER = 1:30
       if N1 == 1 && N2 == 1 
          if STAB 

#             Use direct recursion.

             for JJ = L1:LSTEP:L2
               J = IND( JJ, P )
               JB = IND( JJ, PB )
               J1 = IND( JJ+ITIND+1, PB )
               Z[1,1] = TL[ 1, 1, J ] * Z[1,1] * TR[ 1, 1, J ] - B[ 1, 1, JB ] 
               X[ 1, 1, J1 ] = Z[ 1, 1 ]
             end
          else

#             Use inverse recursion.

             for JJ = L1:LSTEP:L2
                  J = IND( JJ, P )
                  JB = IND( JJ, PB )
                  J1 = IND( JJ+ITIND+1, PB )
                  Z[1,1] = (Z[1,1] + B[ 1, 1, JB ] ) / TL[ 1, 1, J ] / TR[ 1, 1, J ]
                  X[ 1, 1, J1 ] = Z[ 1, 1 ]
             end
          end
       elseif N1 == 1 && N2 == 2 
          if STAB 

#              Use direct recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 X11 = TL[ 1, 1, J ]
                 Y11 = TR[ 1, 1, J ]
                 Y22 = TR[ 2, 2, J ]
                 if REV 
                    Y12 = TR[ 1, 2, J ]
                    Y21 = TR[ 2, 1, J ]
                 else
                    Y12 = TR[ 2, 1, J ]
                    Y21 = TR[ 1, 2, J ]
                 end

                 Z[1,1] = X11*Z[1,1]
                 Z[1,2] = X11*Z[1,2]
                 TEMP = Z[1,1]*Y11 + Z[1,2]*Y21
                 Z[1,2] = Z[1,1]*Y12 + Z[1,2]*Y22 - B[ 1, 2, JB ]
                 Z[1,1] = TEMP - B[ 1, 1, JB ]
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 1, 2, J1 ] = Z[ 1, 2 ]
              end
          else

#              Use inverse recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 BTMP[ 1, 1 ] = ( Z[1,1] + B[1,1,JB] ) / TL[ 1, 1, J ]
                 BTMP[ 2, 1 ] = ( Z[1,2] + B[1,2,JB] ) / TL[ 1, 1, J ]
                 ATMP[ 1, 1 ] = TR[ 1, 1, J ]
                 ATMP[ 2, 2 ] = TR[ 2, 2, J ]
                 if REV 
                    ATMP[ 1, 2 ] = TR[ 2, 1, J ]
                    ATMP[ 2, 1 ] = TR[ 1, 2, J ]
                 else
                    ATMP[ 1, 2 ] = TR[ 1, 2, J ]
                    ATMP[ 2, 1 ] = TR[ 2, 1, J ]
                 end

                 #CALL DGESV( 2, 1, ATMP, 2, JPIV, BTMP, 2, INFO )
                 luslv!(ATMP,view(BTMP,1:2,1:1)) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
                 #BTMP = ATMP\BTMP
                 Z[ 1, 1 ] = BTMP[ 1, 1 ]
                 Z[ 1, 2 ] = BTMP[ 2, 1 ]
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 1, 2, J1 ] = Z[ 1, 2 ]
              end
          end
       elseif N1 == 2 && N2 == 1 
          if STAB 

#              Use direct recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 X11 = TL[ 1, 1, J ]
                 X22 = TL[ 2, 2, J ]
                 if REV 
                    X12 = TL[ 2, 1, J ]
                    X21 = TL[ 1, 2, J ]
                 else
                    X12 = TL[ 1, 2, J ]
                    X21 = TL[ 2, 1, J ]
                 end
                 Y11 = TR[ 1, 1, J ]
 
                 TEMP = X11*Z[1,1] + X12*Z[2,1]
                 Z[2,1] = X21*Z[1,1] + X22*Z[2,1]
                 Z[1,1] = TEMP
                 Z[1,1] = Z[1,1]*Y11 - B[ 1, 1, JB ]
                 Z[2,1] = Z[2,1]*Y11 - B[ 2, 1, JB ]
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 2, 1, J1 ] = Z[ 2, 1 ]
              end
          else

#              Use inverse recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 Z[ 1, 1 ] = ( Z[1,1] + B[1,1,JB] ) / TR[ 1, 1, J ]
                 Z[ 2, 1 ] = ( Z[2,1] + B[2,1,JB] ) / TR[ 1, 1, J ]
                 ATMP[ 1, 1 ] = TL[ 1, 1, J ]
                 ATMP[ 2, 2 ] = TL[ 2, 2, J ]
                 if REV 
                    ATMP[ 1, 2 ] = TL[ 2, 1, J ]
                    ATMP[ 2, 1 ] = TL[ 1, 2, J ]
                 else
                    ATMP[ 1, 2 ] = TL[ 1, 2, J ]
                    ATMP[ 2, 1 ] = TL[ 2, 1, J ]
                 end

                 #CALL DGESV( 2, 1, ATMP, 2, JPIV, Z, 2, INFO )
                 luslv!(ATMP,view(Z,1:2,1:1)) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
                 #Z = ATMP\Z
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 2, 1, J1 ] = Z[ 2, 1 ]
              end
          end
       elseif N1 == 2 && N2 == 2 
          if STAB 

#              Use direct recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 X11 = TL[ 1, 1, J ]
                 X22 = TL[ 2, 2, J ]
                 Y11 = TR[ 1, 1, J ]
                 Y22 = TR[ 2, 2, J ]
                 if REV 
                    X12 = TL[ 2, 1, J ]
                    X21 = TL[ 1, 2, J ]
                    Y12 = TR[ 1, 2, J ]
                    Y21 = TR[ 2, 1, J ]
                 else
                    X12 = TL[ 1, 2, J ]
                    X21 = TL[ 2, 1, J ]
                    Y12 = TR[ 2, 1, J ]
                    Y21 = TR[ 1, 2, J ]
                 end

                 TEMP = X11*Z[1,1] + X12*Z[2,1]
                 Z[2,1] = X21*Z[1,1] + X22*Z[2,1]
                 Z[1,1] = TEMP
                 TEMP = X11*Z[1,2] + X12*Z[2,2]
                 Z[2,2] = X21*Z[1,2] + X22*Z[2,2]
                 Z[1,2] = TEMP
                 TEMP = Z[1,1]*Y11 + Z[1,2]*Y21
                 Z[1,2] = Z[1,1]*Y12 + Z[1,2]*Y22 - B[ 1, 2, JB ]
                 Z[1,1] = TEMP - B[ 1, 1, JB ]
                 TEMP = Z[2,1]*Y11 + Z[2,2]*Y21
                 Z[2,2] = Z[2,1]*Y12 + Z[2,2]*Y22 - B[ 2, 2, JB ]
                 Z[2,1] = TEMP - B[ 2, 1, JB ]
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 2, 1, J1 ] = Z[ 2, 1 ]
                 X[ 1, 2, J1 ] = Z[ 1, 2 ]
                 X[ 2, 2, J1 ] = Z[ 2, 2 ]
              end
          else

#              Use inverse recursion.

             for JJ = L1:LSTEP:L2
                 J = IND( JJ, P )
                 JB = IND( JJ, PB )
                 J1 = IND( JJ+ITIND+1, PB )
                 BTMP[ 1, 1 ] = Z[ 1, 1 ] + B[ 1, 1, JB ]
                 BTMP[ 2, 1 ] = Z[ 1, 2 ] + B[ 1, 2, JB ]
                 BTMP[ 1, 2 ] = Z[ 2, 1 ] + B[ 2, 1, JB ]
                 BTMP[ 2, 2 ] = Z[ 2, 2 ] + B[ 2, 2, JB ]
                 ATMP[ 1, 1 ] = TR[ 1, 1, J ]
                 ATMP[ 2, 2 ] = TR[ 2, 2, J ]
                 if REV 
                    ATMP[ 1, 2 ] = TR[ 2, 1, J ]
                    ATMP[ 2, 1 ] = TR[ 1, 2, J ]
                 else
                    ATMP[ 1, 2 ] = TR[ 1, 2, J ]
                    ATMP[ 2, 1 ] = TR[ 2, 1, J ]
                 end

                 #CALL DGESV( 2, 2, ATMP, 2, JPIV, BTMP, 2, INFO )
                 luslv!(ATMP,view(BTMP,1:2,1:2)) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
                 #BTMP = ATMP\BTMP
                 Z[ 1, 1 ] = BTMP[ 1, 1 ]
                 Z[ 1, 2 ] = BTMP[ 2, 1 ]
                 Z[ 2, 1 ] = BTMP[ 1, 2 ]
                 Z[ 2, 2 ] = BTMP[ 2, 2 ]

                 ATMP[ 1, 1 ] = TL[ 1, 1, J ]
                 ATMP[ 2, 2 ] = TL[ 2, 2, J ]
                 if REV 
                    ATMP[ 1, 2 ] = TL[ 2, 1, J ]
                    ATMP[ 2, 1 ] = TL[ 1, 2, J ]
                 else
                    ATMP[ 1, 2 ] = TL[ 1, 2, J ]
                    ATMP[ 2, 1 ] = TL[ 2, 1, J ]
                 end

                 #CALL DGESV( 2, 2, ATMP, 2, JPIV, Z, 2, INFO )
                 luslv!(ATMP,view(Z,1:2,1:2)) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
                 #Z = ATMP\Z
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 1, 2, J1 ] = Z[ 1, 2 ]
                 X[ 2, 1, J1 ] = Z[ 2, 1 ]
                 X[ 2, 2, J1 ] = Z[ 2, 2 ]
              end
          end
       end
       any(!isfinite,X) && throw("PS:SingularException: A has characteristic multipliers α and β such that αβ ≈ 1")
       DNORM = zero(T)
       for J = 1:N2
          for I = 1:N1
             DNORM = max( DNORM, abs( ZOLD[I,J] - Z[I,J] ) )
             ZOLD[ I, J ] = Z[ I, J ]
          end
       end
       # println("XNORM = $XNORM DNORM = $DNORM")
       DNORM <= EPSM*XNORM && (return -X)
   end
   @warn "iterative process not converged in 30 iterations: solution may be inaccurate"
   return -X
#   END of DPSYLV2
end
@inline function luslv!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
   #
   #  fail = luslv!(A,B)
   #
   # This function is a speed-oriented implementation of a Gaussion-elimination based
   # solver of small order linear equations of the form A*X = B. The computed solution X
   # overwrites the vector B, while the resulting A contains in its upper triangular part,
   # the upper triangular factor U of its LU decomposition.
   # The diagnostic output parameter fail, of type Bool, is set to false in the case
   # of normal return or is set to true if the exact singularity of A is detected
   # or if the resulting B has non-finite components.
   #
   n, m = size(B)
   @inbounds begin
         for k = 1:n
            # find index max
            kp = k
            if k < n
                amax = abs(A[k, k])
                for i = k+1:n
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            iszero(A[kp,k]) && return true
            if k != kp
               # Interchange
               for i = 1:n
                   tmp = A[k,i]
                   A[k,i] = A[kp,i]
                   A[kp,i] = tmp
               end
               for j = 1:m 
                   tmp = B[k,j]
                   B[k,j] = B[kp,j]
                   B[kp,j] = tmp
               end
            end
            # Scale first column
            Akkinv = inv(A[k,k])
            i1 = k+1:n
            Ak = view(A,i1,k)
            rmul!(Ak,Akkinv)
            # Update the rest of A and B
            for j = k+1:n
                axpy!(-A[k,j],Ak,view(A,i1,j))
            end
            for j = 1:m 
                axpy!(-B[k,j],Ak,view(B,i1,j))
            end
         end
         ldiv!(UpperTriangular(A), view(B,:,:))
         return any(!isfinite, B)
   end
end
for PM in (:PeriodicArray, :PeriodicMatrix)
   @eval begin
      function pdplyap(A::$PM, C::$PM; adj::Bool = true, rtol = eps(float(promote_type(eltype(A), eltype(C))))^0.75) 
         A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
         period = promote_period(A, C)
         na = rationalize(period/A.period).num
         K = na*A.nperiod*A.dperiod
         U = psplyapd(A.M, C.M; adj, rtol)
         p = lcm(length(A),length(C))
         return $PM(U, period; nperiod = div(K,p))
      end
   end
end
"""
    pdplyap(A, C; adj = true) -> U

Compute the upper triangular factor `U` of the solution `X = U'U` of the 
periodic discrete-time Lyapunov matrix equation

      A'σXA + C'C = X, if adj = true,

or of the solution `X = UU'` of the periodic discrete-time Lyapunov matrix equation

      AXA' + CC' =  σX, if adj = false, 

where `σ` is the forward shift operator `σX(i) = X(i+1)`. 
The periodic matrix `A` must be stable, i.e., have all characteristic multipliers 
with moduli less than one. 

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods. 
The resulting upper triangular periodic matrix `U` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

The iterative method (Algorithm 5) of [1] and its dual version are employed.  

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
pdplyap(A::PeriodicArray, C::PeriodicArray; adj::Bool = true) 

"""
     psplyapd(A, C; adj = true, rtol = ϵ^(3/4)) -> U

Compute the upper triangular factor `U` of the solution `X = U'U` of the 
periodic discrete-time Lyapunov matrix equation

      A'σXA + C'C = X, if adj = true,

or of the solution `X = UU'` of the periodic discrete-time Lyapunov matrix equation

      AXA' + CC' =  σX, if adj = false, 

where `σ` is the forward shift operator `σX(i) = X(i+1)`. 
The periodic matrix `A` must be stable, i.e., have all characteristic multipliers 
with moduli less than one. 

The periodic matrices `A` and `C` are either stored as 3-dimensional arrays or as
as vectors of matrices. 

The iterative method (Algorithm 5) of [1] and its dual version are employed.  

_Reference:_

[1] A. Varga, "Periodic Lyapunov equations: some applications and new algorithms", 
    Int. J. Control, vol. 67, pp. 69-87, 1997.
"""
function psplyapd(A::AbstractVector{Matrix{T1}}, C::AbstractVector{Matrix{T2}}; adj::Bool = true, rtol = eps(float(promote_type(T1, T2)))^0.75) where {T1, T2}
   pa = length(A) 
   pc = length(C)
   ma, na = size.(A,1), size.(A,2) 
   mc, nc = size.(C,1), size.(C,2) 
   p = lcm(pa,pc)
   all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   if adj
      all([nc[mod(i-1,pc)+1]  == na[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   else
      all([mc[mod(i-1,pc)+1] == ma[mod(i-1,pa)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   end
   rev(t) = reverse(reverse(t,dims=1),dims=2)

   T = promote_type(T1, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
 
   ii = argmin(na)

   nmin = na[ii]
   x = zeros(T,nmin,nmin)

   if adj
      U = Vector{Matrix}(undef,p)
      # compute an initializing periodic factor U[ii]
      V = zeros(T,nmin,nmin)
      Φ = Matrix{T}(I(nmin))
      for i = ii:ii+p-1
         ia = mod(i-1,pa)+1
         ic = mod(i-1,pc)+1
         V = qr([V; C[ic]*Φ]).R
         Φ = A[ia]*Φ
      end
      U[ii] = plyapd(Φ', V')
      for iter = 1:100
          if iter > 1
             y = U[ii]'*U[ii]
             if norm(y-x,1) <= rtol*norm(y)
                break
             end
             x = copy(y)
          end  
          for i = ii+p:-1:ii+1
              j = mod(i-1,p)+1
              jm1 = mod(i-2,p)+1
              iam1 = mod(i-2,pa)+1
              icm1 = mod(i-2,pc)+1
              U[jm1] = qr([U[j]*A[iam1]; C[icm1]]).R
          end
     end
   else
      U = Vector{Matrix}(undef,p)
      # compute an initializing periodic factor U[ii]
      V = zeros(T,nmin,nmin)
      Φ = Matrix{T}(I(nmin))
      for i = ii+p-1:-1:ii
         ia = mod(i-1,pa)+1
         ic = mod(i-1,pc)+1
         V = rev(qr(rev([V Φ*C[ic]]')).R')
         Φ = Φ*A[ia]
      end
      U[ii] = plyapd(Φ, V)
      for iter = 1:100
          if iter > 1
             y = U[ii]*U[ii]'
             if norm(y-x,1) <= rtol*norm(y)
                 break
             end
             x = copy(y)
          end  
          for i = ii+1:ii+p
              j = mod(i-1,p)+1
              jm1 = mod(i-2,p)+1
              iam1 = mod(i-2,pa)+1
              icm1 = mod(i-2,pc)+1
              U[j] = rev(qr(rev([A[iam1]*U[jm1] C[icm1]]')).R')
          end
     end
   end

   for i = 1:p
       U[i] = makesp(U[i];adj)
   end
   return U
end

function makesp(r; adj = true)
   # make upper trapezoidal matrix square and with positive diagonal elements
   if adj
      n, nc = size(r)
      rp = [r; zeros(nc-n,nc)]
      for i = 1:n
          rp[i,i] < 0 && (rp[i,i:end] = -rp[i,i:end])
      end
   else
      n, nc = size(r)
      rp = [zeros(n,n-nc) r ]
      for i = n-nc+1:n
          rp[i,i] < 0 && (rp[1:i,i] = -rp[1:i,i])
      end
   end
   return rp
end
function psplyapd(A::AbstractArray{T1,3}, C::AbstractArray{T2,3}; adj::Bool = true, rtol = eps(float(promote_type(T1, T2)))^0.75) where {T1, T2}
   mc, nc, pc = size(C)
   n = LinearAlgebra.checksquare(A[:,:,1]) 
   pa = size(A,3)
   p = lcm(pa,pc)
   if adj
      nc  == n || throw(DimensionMismatch("incompatible dimensions between A and C"))
   else
      mc == n ||  throw(DimensionMismatch("incompatible dimensions between A and C"))
   end
   rev(t) = reverse(reverse(t,dims=1),dims=2)

   T = promote_type(T1, T2)
   T <: BlasFloat  || (T = promote_type(Float64,T))
 
   x = zeros(T,n,n)
   U = Array{T,3}(undef,n,n,p)

   if adj
      # compute an initializing periodic factor U[ii]
      V = zeros(T,n,n)
      Φ = Matrix{T}(I(n))
      for i = 1:p
          ia = mod(i-1,pa)+1
          ic = mod(i-1,pc)+1
          V = qr([V; C[:,:,ic]*Φ]).R
          Φ = A[:,:,ia]*Φ
      end
      U[:,:,1] = plyapd(Φ', V')
      for iter = 1:100
          if iter > 1
             y = U[:,:,1]'*U[:,:,1]
             if norm(y-x,1) <= rtol*norm(y)
                break
             end
             x = copy(y)
          end  
          for i = p+1:-1:2
              j = mod(i-1,p)+1
              jm1 = mod(i-2,p)+1
              iam1 = mod(i-2,pa)+1
              icm1 = mod(i-2,pc)+1
              U[:,:,jm1] = qr([U[:,:,j]*A[:,:,iam1]; C[:,:,icm1]]).R
          end
      end
   else
      # compute an initializing periodic factor U[ii]
      V = zeros(T,n,n)
      Φ = Matrix{T}(I(n))
      for i = p:-1:1
         ia = mod(i-1,pa)+1
         ic = mod(i-1,pc)+1
         V = rev(qr(rev([V Φ*C[:,:,ic]]')).R')
         Φ = Φ*A[:,:,ia]
      end
      U[:,:,1] = plyapd(Φ, V)
      for iter = 1:100
          if iter > 1
             y = U[:,:,1]*U[:,:,1]'
             if norm(y-x,1) <= rtol*norm(y)
                 break
             end
             x = copy(y)
          end  
          for i = 2:p+1
              j = mod(i-1,p)+1
              jm1 = mod(i-2,p)+1
              iam1 = mod(i-2,pa)+1
              icm1 = mod(i-2,pc)+1
              U[:,:,j] = rev(qr(rev([A[:,:,iam1]*U[:,:,jm1] C[:,:,icm1]]')).R')
          end
     end
   end

   for i = 1:p
       U[:,:,i] = makesp(U[:,:,i];adj)
   end
   return U
end
for PM in (:PeriodicArray, :PeriodicMatrix)
   @eval begin
      function prdplyap(A::$PM, C::$PM) 
         pdplyap(A, C; adj = true)
      end
      function prdplyap(A::$PM, C::AbstractArray)
         pdplyap(A, $PM(C, A.Ts; nperiod = 1);  adj = true)
      end
      function pfdplyap(A::$PM, C::$PM) 
         pdplyap(A, C; adj = false)
      end
      function pfdplyap(A::$PM, C::AbstractArray)
         pdplyap(A, $PM(C, A.Ts; nperiod = 1); adj = false)
      end
   end
end
"""
    prdplyap(A, C) -> U

Compute the upper triangular factor `U` of the solution `X = U'*U` of the 
reverse time periodic discrete-time Lyapunov matrix equation

    A'σXA + C'C = X

where `σ` is the forward shift operator `σX(i) = X(i+1)`. 
The periodic matrix `A` must be stable, i.e., have all characteristic multipliers 
with moduli less than one. 

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods. 
The resulting upper triangular periodic matrix `U` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly.  

Note: `X` is the _observability Gramian_ of the periodic pair `(A,C)`.              
"""
prdplyap(A::PeriodicArray, C::PeriodicArray) 
"""
    pfdplyap(A, B) -> U

Compute the upper triangular factor `U` of the solution `X = U*U'` of the 
forward-time periodic discrete-time Lyapunov equation

    AXA' + BB' = σX

where `σ` is the forward shift operator `σX(i) = X(i+1)`.  
The periodic matrix `A` must be stable, i.e., have all characteristic multipliers 
with moduli less than one. 
               
The periodic matrices `A` and `B` must have the same type, the same dimensions and commensurate periods. 
The resulting upper triangular periodic matrix `U` has the period 
set to the least common commensurate period of `A` and `B` and the number of subperiods
is adjusted accordingly.  

Note: `X` is the _reachability Gramian_ of the periodic pair `(A,B)`.              
"""
pfdplyap(A::PeriodicArray, B::PeriodicArray) 


