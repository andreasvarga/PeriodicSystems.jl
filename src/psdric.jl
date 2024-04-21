"""
     prdric(A, B, R, Q[, S]; itmax = 0, nodeflate = false, fast, rtol) -> (X, EVALS, F)

Solve the periodic Riccati difference equation

      X(i) = Q(i) + A(i)'X(i+1)A(i) - (A(i)'X(i+1)B(i) + S(i))*
                                     -1
             (B(i)'X(i+1)B(i) + R(i))  (A(i)'X(i+1)B(i) + S(i))' 

and compute the stabilizing periodic state feedback

                                      -1
      F(i) = -(B(i)'X(i+1)B(i) + R(i))  (B(i)'X(i+1)A(i) + S(i)')

and the corresponding stable closed-loop characteristic multipliers of `A(i)-B(i)F(i)` in `EVALS`. 

The `n×n` and `n×m` periodic matrices `A(i)` and `B(i)` are contained in the 
`PeriodicArray` objects `A` and `B`, and must have the same sampling time. 
`R(i)`, `Q(i)` and `S(i)` are `m×m`, `n×n` and `n×m` periodic matrices of same sampling times 
as  `A` and `B`, and such that `R(i)` and `Q(i)` are symmetric. `R(i)`, `Q(i)` and `S(i)` are contained in the 
`PeriodicArray` objects `R`, `Q` and `S`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 
The resulting symmetric periodic solution `X` and periodic state feedback gain `F` have the period 
set to the least common commensurate period of `A`, `B`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 

If `fast = true`, the fast structure exploiting pencil reduction based method of [1] is used
to determine a periodic generator in `X(1)`, which allows to generate iteratively the solution 
over the whole period.  
If `fast = false` (default), the periodic Schur decomposition based approach of [1] is employed, applied to a 
symplectic pair of periodic matrices. If `nodeflate = false` (default), the underlying periodic pencil 
is preprocessed to eliminate (deflate) the infinite characteristic multipliers originating 
from the problem structure. If `nodeflate = true`, no preliminary deflation is performed.

An iterative refining of the accuracy of the computed solution 
can be performed by using `itmax = k`, with `k > 0` (default: `k = 0`). 

To detect singularities of involved matrices, the keyword parameter `rtol = tol` can be used to  
specify the lower bound for the 1-norm reciprocal condition number. 
The default value of  `tol` is `n*ϵ`, where `ϵ` is the working _machine epsilon_.

_References_

[1] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.
    
"""
function prdric(A::PA1, B::PA2, R::PA3, Q::PA4, S::Union{PA5,Missing} = missing; itmax::Int = 0, nodeflate::Bool = false, PSD_SLICOT::Bool = true, fast = false, rtol::Real = size(A,1)*eps()) where 
    {PA1 <: PeriodicArray, PA2 <: PeriodicArray, PA3 <: PeriodicArray, PA4 <: PeriodicArray, PA5 <: PeriodicArray}
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
   m = size(B,2)
   (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
   (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
   issymmetric(R) || error("the periodic matrix R must be symmetric")
   issymmetric(Q) || error("the periodic matrix Q must be symmetric")
   if ismissing(S)
      S = PeriodicArray(zeros(n,m),A.period,nperiod = max(A.dperiod,B.dperiod,Q.dperiod,R.dperiod))
   end 
   A.Ts ≈ B.Ts ≈ R.Ts ≈ Q.Ts || error("A, B, R and Q must have the same sampling time")
   
   pa = size(A.M,3)
   pb = size(B.M,3)
   pq = size(Q.M,3)
   pr = size(R.M,3)
   ps = size(S.M,3)
   p = lcm(pa,pb,pq,pr,ps)
   period = promote_period(A, B, R, Q, S)

   epsm = eps()

   #                   [  A(i)  0     B(i) ]             [   I    0      0   ]
   #  Work on   M(i) = [ -Q(i)  I    -S(i) ] and  L(i) = [   0   A(i)'   0   ]
   #                   [ S(i)'  0     R(i) ]             [   0  -B(i)'   0   ]
   
   n2 = 2*n
   ZERA = zeros(n,n)
   ZERBT = zeros(m,n)
   EYE = eye(Float64,n)
   if fast
      # use fast structure exploiting reduction of the symplectic pencil
      ZERA2 = zeros(n2,n2)
      a = [A.M[:,:,1] ZERA; -Q.M[:,:,1]  EYE; S.M[:,:,1]' ZERBT]
      e = [ EYE ZERA; ZERA A.M[:,:,1]'; ZERBT -B.M[:,:,1]'] 
      QR = qr([B.M[:,:,1]; -S.M[:,:,1]; R.M[:,:,1]])
      i1 = m+1:n2+m; 
      i2 = n2+1:n2+n2
      si = QR.Q[:,i1]'*a
      ti = QR.Q[:,i1]'*e
        for i = 2:p
            (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
            a = [A.M[:,:,ia] ZERA; -Q.M[:,:,iq]  EYE; S.M[:,:,is]' ZERBT]
            e = [ EYE ZERA; ZERA A.M[:,:,ia]'; ZERBT -B.M[:,:,ib]'] 
            QR = qr([B.M[:,:,ib]; -S.M[:,:,is]; R.M[:,:,ir]])
            F = qr([ -ti ; QR.Q[:,i1]'*a ])     
            si = (F.Q'*[si; ZERA2])[i2,:];  
            ti = (F.Q'*[ZERA2; QR.Q[:,i1]'*e])[i2,:]; 
         end
      SF = schur!(ti, si)  # use (e,a) instead (a,e) to favor natural appearance of large eigenvalues
      select = abs.(SF.values) .> 1
      n == count(select .== true) || error("The symplectic pencil is not dichotomic")
      n == count(view(select,1:n)) || ordschur!(SF, select)
      i1 = 1:n; i2 = n+1:n2
      EVALS = SF.values[i2]
      # compute the periodic generator in i = 1
      x = SF.Z[i2,i1]/SF.Z[i1,i1]; x = (x+x')/2
   
      NormRes = 1; it = 0 
      X = Array{Float64,3}(undef, n, n, p)
      F = Array{Float64,3}(undef, m, n, p)
      tol = sqrt(epsm)/100000
      while NormRes > tol && it <= itmax
        for i = p:-1:1
            (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
            F[:,:,i] = (B.M[:,:,ib]'*x*B.M[:,:,ib]+R.M[:,:,ir])\(B.M[:,:,ib]'*x*A.M[:,:,ia]+S.M[:,:,is]')
            x = A.M[:,:,ia]'*x*A.M[:,:,ia] - (A.M[:,:,ia]'*x*B.M[:,:,ib] + S.M[:,:,is])*F[:,:,i] + Q.M[:,:,iq]
            X[:,:,i] = (x+x')/2
        end
        (ia,ib,iq,ir,is) =  mod.(p-1,(pa,pb,pq,pr,ps)).+1
        G = (B.M[:,:,ib]'*x*B.M[:,:,ib]+R.M[:,:,ir])\(B.M[:,:,ib]'*x*A.M[:,:,ia]+S.M[:,:,is]')
        Res = A.M[:,:,ia]'*x*A.M[:,:,ia] - (A.M[:,:,ia]'*x*B.M[:,:,ib] + S.M[:,:,is])*G + Q.M[:,:,iq]- X[:,:,p]
        NormRes = norm(Res) / max(1,norm(X[:,:,p])); 
        #@show it, NormRes
        it += 1
      end
      return PeriodicArray(X,period), EVALS, PeriodicArray(F,period)
    end
    p2 = p+p
    St = Array{Float64,3}(undef, n2+m, n2+m, p2) 
    i1 = 1:n
    i2 = n+1:n2
    iric = 1:n2
    if nodeflate
       # compute eigenvalues of the quotient product 
       X = Array{Float64,3}(undef, n, n, p)
       F = Array{Float64,3}(undef, m, n, p)
       k = 1
       for i = 1:p
           (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
           St[:,:,k] = [ A.M[:,:,ia] ZERA B.M[:,:,ib]; -Q.M[:,:,iq]  EYE -S.M[:,:,is]; S.M[:,:,is]' ZERBT R.M[:,:,ir] ]
           St[:,:,k+1] = [ EYE zeros(n,n+m); ZERA A.M[:,:,ia]' zeros(n,m); ZERBT -B.M[:,:,ib]' zeros(m,m)]
           k += 2
       end  
       s = trues(p2); [s[2*i] = false for i in 1:p]
       _, Zt, ev, sind,  = pgschur!(St, s)
       select = abs.(ev) .< 1 
       n == count(select .== true) || error("The symplectic pencil is not dichotomic")
       pgordschur!(St, s, Zt, select; schurindex = sind)
       EVALS =  ev[select]
       for i = 1:p
           xi = view(Zt,n+1:n2+m,i1,2*i-1)
           FU = MatrixEquations._LUwithRicTest(view(Zt,i1,i1,2*i-1),rtol)
           rdiv!(xi,FU)
           F[:,:,i] = -xi[n+1:end,:]
           x = view(xi,i1,:)
           X[:,:,i] = (x+x')/2
       end 
    else
       k = 1
       for i = 1:p
           (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
           H = qr([A.M[:,:,ia]'; -B.M[:,:,ib]'])
           L2 = H.Q'*[-Q.M[:,:,iq]  EYE -S.M[:,:,is]; S.M[:,:,is]' ZERBT R.M[:,:,ir]]
           St[:,:,k] = [ A.M[:,:,ia] ZERA B.M[:,:,ib]; L2]
           St[:,:,k+1] = [ EYE zeros(n,n+m); ZERA H.R zeros(n,m); zeros(m,n2+m)]
           k += 2
       end  
       k = 1     
       z = Array{Float64,3}(undef, n2+m, n2, p) 
       for i = 1:p
          G = qr(St[n2+1:end,:,k]')
          cond(G.R) * epsm  < 1 || error("The extended symplectic pencil is not regular")
          zi = ((G.Q*I)[:,[m+1:m+n2; 1:m]])[:,iric]
          St[:,iric,k] = St[:,:,k]*zi
          km1 = k == 1 ? p2 : k-1
          St[:,iric, km1] = St[:,:, km1]*zi
          z[:,:,i] = zi
          k += 2
       end
       s = trues(p2); [s[2*i] = false for i in 1:p]
       # compute eigenvalues for the inverse product to exploit the natural order of
       # computed eigenvalues (i.e., large eigenvalues first)
       St, Zt, ev, sind,  = pgschur(view(St,iric,iric,p2:-1:1), s)
       select = abs.(ev) .> 1 
       n == count(select .== true) || error("The symplectic pencil is not dichotomic")
       n == count(view(select,1:n)) || pgordschur!(St, s, Zt, select; schurindex = sind)
       EVALS =  ev[.!select]
       X = Array{Float64,3}(undef, n, n, p)
       F = Array{Float64,3}(undef, m, n, p)
       for i = 1:p
          #zi = z[:,:,i]*Zt[:,i1,2*i-1]
          zi = z[:,:,i]*Zt[:,i1,mod(p2-2*i+2,p2)+1]
          #xi = view(Zt,n+1:n2+m,i1,2*i-1)
          FU = MatrixEquations._LUwithRicTest(view(zi,i1,i1),rtol)
        #   rdiv!(xi,FU)
        #   F[:,:,i] = -xi[n+1:end,:]
        #   X[:,:,i] = xi[i1,:]
          xi = zi[n+1:end,i1]/FU
          F[:,:,i] = -xi[n+1:end,:]
          x = view(xi,i1,:)
          X[:,:,i] = (x+x')/2
       end  
   end 
   if itmax > 0 
      NormRes = 1; it = 0 
      tol = sqrt(epsm)/100000
      x = X[:,:,1]
      while NormRes > tol && it < itmax
       for i = p:-1:1
           (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
           F[:,:,i] = (B.M[:,:,ib]'*x*B.M[:,:,ib]+R.M[:,:,ir])\(B.M[:,:,ib]'*x*A.M[:,:,ia]+S.M[:,:,is]')
           x = A.M[:,:,ia]'*x*A.M[:,:,ia] - (A.M[:,:,ia]'*x*B.M[:,:,ib] + S.M[:,:,is])*F[:,:,i] + Q.M[:,:,iq]
           X[:,:,i] = (x+x')/2
       end
       (ia,ib,iq,ir,is) =  mod.(p-1,(pa,pb,pq,pr,ps)).+1
       G = (B.M[:,:,ib]'*x*B.M[:,:,ib]+R.M[:,:,ir])\(B.M[:,:,ib]'*x*A.M[:,:,ia]+S.M[:,:,is]')
       Res = A.M[:,:,ia]'*x*A.M[:,:,ia] - (A.M[:,:,ia]'*x*B.M[:,:,ib] + S.M[:,:,is])*G + Q.M[:,:,iq]- X[:,:,p]
       NormRes = norm(Res) / max(1,norm(X[:,:,p])); 
       #@show it, NormRes
       it += 1
      end
  end
  return PeriodicArray(X,period), EVALS, PeriodicArray(F,period)
end
function prdric(A::PM1, B::PM2, R::PM3, Q::PM4, S::AbstractMatrix; kwargs...) where 
   {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <: PeriodicArray, PM4 <: PeriodicArray}
   nperiod = rationalize(A.period/A.Ts).num
   prdric(A, B, R, Q, PeriodicArray(S, A.period; nperiod); kwargs...)
end
function prdric(A::PM1, B::PM2, R::AbstractMatrix, Q::PM4, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM4 <: PeriodicArray, PM5 <: PeriodicArray}
  nperiod = rationalize(A.period/A.Ts).num
  prdric(A, B, PeriodicArray(R, A.period; nperiod), Q, ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicArray(S, A.period; nperiod) : S; kwargs...)
end
function prdric(A::PM1, B::PM2, R::Union{PM4,AbstractMatrix}, Q::AbstractMatrix, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM4 <: PeriodicArray, PM5 <: PeriodicArray}
  nperiod = rationalize(A.period/A.Ts).num
  prdric(A, B, isa(R,AbstractMatrix) ? PeriodicArray(R, A.period; nperiod) : R, PeriodicArray(Q, A.period; nperiod), ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicArray(S, A.period; nperiod) : S; kwargs...)
end
"""
     prdric(A, B, R, Q[, S]; itmax = 0, nodeflate = false, fast, rtol) -> (X, EVALS, F)

Solve the periodic Riccati difference equation

      X(i) = Q(i) + A(i)'X(i+1)A(i) - (A(i)'X(i+1)B(i) + S(i))*
                                     -1
             (B(i)'X(i+1)B(i) + R(i))  (A(i)'X(i+1)B(i) + S(i))' 

and compute the stabilizing periodic state feedback

                                      -1
      F(i) = -(B(i)'X(i+1)B(i) + R(i))  (B(i)'X(i+1)A(i) + S(i)')

and the corresponding stable closed-loop _core_ characteristic multipliers of `A(i)-B(i)F(i)` in `EVALS`. 

The `n(i+1)×n(i)` and `n(i+1)×m` periodic matrices `A(i)` and `B(i)` are contained in the 
`PeriodicMatrix` objects `A` and `B`, and must have the same sampling time. 
`R(i)`, `Q(i)` and `S(i)` are `m×m`, `n(i)×n(i)` and `n(i)×m` periodic matrices of same sampling times 
as  `A` and `B`, and such that `R(i)` and `Q(i)` are symmetric. `R(i)`, `Q(i)` and `S(i)` are contained in the 
`PeriodicMatrix` objects `R`, `Q` and `S`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 
The resulting `n(i)×n(i)` symmetric periodic solution `X(i)` and `m×n(i)` periodic state feedback gain `F(i)` have the period 
set to the least common commensurate period of `A`, `B`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 

If `fast = true`, the fast structure exploiting pencil reduction based method of [1] is used
to determine a periodic generator in `X(j)`, which allows to generate iteratively the solution 
over the whole period. The value of `j` corresponds to the least dimension `nc` of `n(i)` 
(which is also the number of core characteristic multipliers). 
If `fast = false` (default), the periodic Schur decomposition based approach of [1] is employed, applied to a 
symplectic pair of periodic matrices. If `nodeflate = false` (default), the underlying periodic pencil 
is preprocessed to eliminate (deflate) the infinite characteristic multipliers originating 
from the problem structure. If `nodeflate = true`, no preliminary deflation is performed.

An iterative refining of the accuracy of the computed solution 
can be performed by using `itmax = k`, with `k > 0` (default: `k = 0`). 

To detect singularities of involved matrices, the keyword parameter `rtol = tol` can be used to  
specify the lower bound for the 1-norm reciprocal condition number. 
The default value of  `tol` is `n*ϵ`, where `ϵ` is the working _machine epsilon_ and `n` is the maximum of `n(i)`.

_References_

[1] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.   
"""
function prdric(A::PM1, B::PM2, R::PM3, Q::PM4, S::Union{PM5,Missing} = missing; itmax::Int = 0, nodeflate::Bool = false, PSD_SLICOT::Bool = true, fast = false, rtol::Real = size(A.M[1],1)*eps()) where 
  {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  ma, na = size(A) 
  mb, nb = size(B) 
  nq = size(Q,2)
  missingS = ismissing(S)
  missingS || (ms = size(S,1))
  m = maximum(nb)
  m == minimum(nb) || throw(DimensionMismatch("only constant number of columns of B supported"))
  pr = length(R) 
  pa = length(A) 
  pb = length(B) 
  pq = length(Q) 
  ps = missingS ? 1 : length(S)
  p = lcm(pa,pb,pq,pr,ps)

  # perform checks applicable to both constant and time-varying dimensions  
  all(view(ma,mod.(1:p-1,pa).+1) .== view(mb,mod.(1:p-1,pb).+1)) || 
     error("the number of rows of A[i] must be equal to the number of rows of B[i]")

  all([LinearAlgebra.checksquare(R.M[i]) == m for i in 1:pr]) ||
          throw(DimensionMismatch("incompatible dimensions between B and R"))
  all([issymmetric(R.M[i]) for i in 1:pr]) || error("all R[i] must be symmetric matrices")
  all([issymmetric(Q.M[i]) for i in 1:pq]) || error("all Q[i] must be symmetric matrices")
  n = maximum(na) 
  ma1 = maximum(ma) 
  constdim = (n == minimum(na) && ma1 == minimum(ma))
  if constdim 
     # constant dimensions
     n == ma1 || error("the periodic matrix A must be square")
     n == minimum(nq) == maximum(nq) || throw(DimensionMismatch("incompatible dimensions between A and Q"))
     missingS && 
        (S = PeriodicMatrix(zeros(n,m),A.period,nperiod = rationalize(A.period/A.Ts).num))
     ms = size(S,1)
  else
     # time-varying dimensions
     missingS && (ps = p; S = PeriodicMatrix([zeros(na[i],m) for i in 1:p], A.period, nperiod = 1))
     pa == pb == pq == ps || throw(DimensionMismatch("A, B, Q and S must have the same length"))
     (pr == 1 || pr == pa) || throw(DimensionMismatch("R must have the length equal to or length of A"))
     all(ma .== view(na,mod.(1:pa,pa).+1)) || 
         error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
     all([LinearAlgebra.checksquare(Q.M[i]) == na[i] for i in 1:p]) ||
         throw(DimensionMismatch("incompatible dimensions between A and Q"))
  end
  A.Ts ≈ B.Ts ≈ R.Ts ≈ Q.Ts ≈ S.Ts || error("A, B, R, Q and S must have the same sampling time")

  period = promote_period(A, B, R, Q, S)

  epsm = eps()
 
  #                   [  A(i)  0     B(i) ]             [   I    0      0   ]
  #  Work on   M(i) = [ -Q(i)  I    -S(i) ] and  L(i) = [   0   A(i)'   0   ]
  #                   [ S(i)'  0     R(i) ]             [   0  -B(i)'   0   ]
                                         
  n2 = 2*n
  ZERA = zeros(n,n)
  ZERBT = zeros(m,n)
  EYE = eye(Float64,n)
  if fast
     # use fast structure exploiting reduction of the symplectic periodic pair (M(i),L(i))
     ZERA2 = zeros(n2,n2)
     k = argmin(na)
     ni = na[k]
     nip1 = na[mod(k,pa)+1]
     (ia,ib,iq,ir,is) =  mod.(k-1,(pa,pb,pq,pr,ps)).+1
     a = [A.M[ia] view(ZERA,1:nip1,1:ni); -Q.M[iq]  view(EYE,1:ni,1:ni); S.M[is]' view(ZERBT,1:m,1:ni)]
     e = [ view(EYE,1:nip1,1:nip1) view(ZERA,1:nip1,1:nip1); view(ZERA,1:ni,1:nip1) A.M[ia]'; view(ZERBT,1:m,1:nip1) -B.M[ib]'] 
     QR = qr([B.M[ib]; -S.M[is]; R.M[ir]])
     i1 = m+1:nip1+ni+m; 
     si = QR.Q[:,i1]'*a
     ti = QR.Q[:,i1]'*e
     n1 = size(si,2)
     nc = na[k]
     for i = k+1:p+k-1 
         (ia,ib,iq,ir,is) =  mod.(i-1,(pa,pb,pq,pr,ps)).+1
         ni = na[ia]
         nip1 = na[mod(ia,pa)+1]
         a = [A.M[ia] view(ZERA,1:nip1,1:ni); -Q.M[iq]  view(EYE,1:ni,1:ni); S.M[is]' view(ZERBT,1:m,1:ni)]
         e = [ view(EYE,1:nip1,1:nip1) view(ZERA,1:nip1,1:nip1); view(ZERA,1:ni,1:nip1) A.M[ia]'; view(ZERBT,1:m,1:nip1) -B.M[ib]'] 
         QR = qr([B.M[ib]; -S.M[is]; R.M[ir]])
         i1 = m+1:nip1+ni+m; 
         F = qr([ -ti ; QR.Q[:,i1]'*a]) 
         ni2 = 2*na[ia]
         si = (F.Q'*[si; view(ZERA2,1:ma[ia]+na[ia],1:n1)])[ni2+1:end,:]  
         ip1 = ia+1; ip1 > pa && (ip1 = 1)
         ti = (F.Q'*[view(ZERA2,1:size(ti,1),1:2*na[ip1]); QR.Q[:,i1]'*e])[ni2+1:end,:]
     end

     SF = schur!(ti, si)  # use (e,a) instead (a,e) to favor natural appearance of large eigenvalues
     select = abs.(SF.values) .> 1
     nc == count(select .== true) || error("The symplectic pencil is not dichotomic")
     nc == count(view(select,1:nc)) || ordschur!(SF, select)
     i1 = 1:nc; i2 = nc+1:2*nc
     EVALS = SF.values[i2]
     # compute the periodic generator in i = 1
     x = SF.Z[i2,i1]/SF.Z[i1,i1]; x = (x+x')/2
        
     NormRes = 1; it = 0 
     X = Vector{Array{Float64}}(undef, p)
     F = Vector{Array{Float64}}(undef, p)
     tol = sqrt(epsm)/100000
     while NormRes > tol && it <= itmax
       for i = p:-1:1
           ix = mod(i+k-2,p)+1
           (ia,ib,iq,ir,is) =  mod.(i+k-2,(pa,pb,pq,pr,ps)).+1
           F[ix] = (B.M[ib]'*x*B.M[ib]+R.M[ir])\(B.M[ib]'*x*A.M[ia]+S.M[is]')
           x = A.M[ia]'*x*A.M[ia] - (A.M[ia]'*x*B.M[ib] + S.M[is])*F[ix] + Q.M[iq]
           X[ix] = (x+x')/2
       end
       (ia,ib,iq,ir,is) =  mod.(p+k-1-1,(pa,pb,pq,pr,ps)).+1
       G = (B.M[ib]'*x*B.M[ib]+R.M[ir])\(B.M[ib]'*x*A.M[ia]+S.M[is]')
       kp = mod(p+k-2,p)+1
       Res = A.M[ia]'*x*A.M[ia] - (A.M[ia]'*x*B.M[ib] + S.M[is])*G + Q.M[iq]- X[kp]
       NormRes = norm(Res) / max(1,norm(X[kp])); 
       #@show it, NormRes
       it += 1
     end
     return PeriodicMatrix(X,period), EVALS, PeriodicMatrix(F,period)
  else
    At = zeros(n,n,pa); [copyto!(view(At,1:ma[i],1:na[i],i),A.M[i]) for i in 1:pa]
    Bt = zeros(n,m,pb); [copyto!(view(Bt,1:mb[i],1:m,i),B.M[i]) for i in 1:pb]
    Rt = zeros(m,m,pr); [copyto!(view(Rt,1:m,1:m,i),R.M[i]) for i in 1:pr]
    Qt = zeros(n,n,pq); [copyto!(view(Qt,1:nq[i],1:nq[i],i),Q.M[i]) for i in 1:pq]
    St = zeros(n,m,ps); missingS || [copyto!(view(St,1:ms[i],1:m,i),S.M[i]) for i in 1:ps]
    Xt, EVALS, Ft = prdric(PeriodicArray(At,A.period;nperiod = A.nperiod),PeriodicArray(Bt,B.period;nperiod = B.nperiod),
                           PeriodicArray(Rt,R.period;nperiod = R.nperiod),PeriodicArray(Qt,Q.period;nperiod = Q.nperiod),
                           missingS ? missing : PeriodicArray(St,S.period;nperiod = S.nperiod); itmax, nodeflate, PSD_SLICOT, fast, rtol)

    return PeriodicMatrix([Xt.M[1:na[mod(i-1,pa)+1],1:na[mod(i-1,pa)+1],i] for i in 1:p],period), EVALS, PeriodicMatrix([Ft.M[1:m,1:na[mod(i-1,pa)+1],i] for i in 1:p],period)
  end
   
end
function prdric(A::PM1, B::PM2, R::PM3, Q::PM4, S::AbstractMatrix; kwargs...) where 
   {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix, PM4 <: PeriodicMatrix}
   nperiod = rationalize(A.period/A.Ts).num
   prdric(A, B, R, Q, PeriodicMatrix(S, A.period; nperiod); kwargs...)
end
function prdric(A::PM1, B::PM2, R::AbstractMatrix, Q::PM4, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  nperiod = rationalize(A.period/A.Ts).num
  prdric(A, B, PeriodicMatrix(R, A.period; nperiod), Q, ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicMatrix(S, A.period; nperiod) : S; kwargs...)
end
function prdric(A::PM1, B::PM2, R::Union{PM4,AbstractMatrix}, Q::AbstractMatrix, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  ma, na = size(A)
  if maximum(na) == minimum(na) && maximum(ma) == minimum(ma)
     nperiod = rationalize(A.period/A.Ts).num
     prdric(A, B, isa(R,AbstractMatrix) ? PeriodicMatrix(R, A.period; nperiod) : R, PeriodicMatrix(Q, A.period; nperiod), ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicMatrix(S, A.period; nperiod) : S; kwargs...)
  else
     error("Constant Q and S are not possible for time-varying dimensions")
  end
end
"""
     pfdric(A, C, R, Q[, S]; itmax = 0, nodeflate = false, fast, rtol) -> (X, EVALS, F)

Solve the periodic Riccati difference equation

      X(i+1) = Q(i) + A(i)X(i)A(i)' - (A(i)X(i)C(i)' + S(i))*
                                     -1
               (C(i)X(i)C(i)' + R(i))  (A(i)X(i)C(i)' + S(i))' 

and compute the stabilizing periodic Kalman gain

                                                           -1
      F(i) = -(C(i)X(i)A(i)' + S(i)')(C(i)X(i)C(i)' + R(i))  

and the corresponding stable Kalman filter characteristic multipliers of `A(i)-F(i)C(i)` in `EVALS`. 

The `n×n` and `m×n` periodic matrices `A(i)` and `C(i)` are contained in the 
`PeriodicArray` objects `A` and `C`, and must have the same sampling time. 
`R(i)`, `Q(i)` and `S(i)` are `m×m`, `n×n` and `m×n` periodic matrices of same sampling times 
as  `A` and `C`, and such that `R(i)` and `Q(i)` are symmetric. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 
The resulting symmetric periodic solution `X` and Kalman filter gain `F` have the period 
set to the least common commensurate period of `A`, `C`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 

The dual method of [1] is employed (see [`prdric`](@ref) for the description of keyword parameters).

_References_

[1] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.
    
"""
function pfdric(A::PA1, C::PA2, R::PA3, Q::PA4, S::Union{PA5,Missing} = missing; kwargs...) where 
  {PA1 <: PeriodicArray, PA2 <: PeriodicArray, PA3 <: PeriodicArray, PA4 <: PeriodicArray, PA5 <: PeriodicArray}
  Xt, EVALS, Ft =  prdric(reverse(A)', reverse(C)', reverse(R), reverse(Q), ismissing(S) ? S : reverse(S); kwargs...) 
  return reverse(pmshift(Xt)), EVALS, reverse(Ft)'
end
function pfdric(A::PM1, C::PM2, R::PM3, Q::PM4, S::AbstractMatrix; kwargs...) where 
   {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <: PeriodicArray, PM4 <: PeriodicArray}
   nperiod = rationalize(A.period/A.Ts).num
   pfdric(A, C, R, Q, PeriodicArray(S, A.period; nperiod); kwargs...)
end
function pfdric(A::PM1, C::PM2, R::AbstractMatrix, Q::PM4, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
   {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM4 <: PeriodicArray, PM5 <: PeriodicArray}
   nperiod = rationalize(A.period/A.Ts).num
   pfdric(A, C, PeriodicArray(R, A.period; nperiod), Q, ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicArray(S, A.period; nperiod) : S; kwargs...)
end
function pfdric(A::PM1, C::PM2, R::Union{PM4,AbstractMatrix}, Q::AbstractMatrix, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
   {PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM4 <: PeriodicArray, PM5 <: PeriodicArray}
   nperiod = rationalize(A.period/A.Ts).num
   pfdric(A, C, isa(R,AbstractMatrix) ? PeriodicArray(R, A.period; nperiod) : R, PeriodicArray(Q, A.period; nperiod), ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicArray(S, A.period; nperiod) : S; kwargs...)
end
"""
     pfdric(A, C, R, Q[, S]; itmax = 0, nodeflate = false, fast, rtol) -> (X, EVALS, F)

Solve the periodic Riccati difference equation

      X(i+1) = Q(i) + A(i)X(i)A(i)' - (A(i)X(i)C(i)' + S(i))*
                                     -1
               (C(i)X(i)C(i)' + R(i))  (A(i)X(i)C(i)' + S(i))' 

and compute the stabilizing periodic Kalman gain

                                                           -1
      F(i) = -(C(i)X(i)A(i)' + S(i)')(C(i)X(i)C(i)' + R(i))  

and the corresponding stable Kalman filter _core_ characteristic multipliers of `A(i)-F(i)C(i)` in `EVALS`. 

The `n(i+1)×n(i)` and `m×n(i)` periodic matrices `A(i)` and `C(i)` are contained in the 
`PeriodicMatrix` objects `A` and `C`, and must have the same sampling time. 
`R(i)`, `Q(i)` and `S(i)` are `m×m`, `n(i)×n(i)` and `m×n(i)` periodic matrices of same sampling times 
as  `A` and `C`, and such that `R(i)` and `Q(i)` are symmetric. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 
The resulting symmetric periodic solution `X` and Kalman filter gain `F` have the period 
set to the least common commensurate period of `A`, `C`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 

The dual method of [1] is employed (see [`prdric`](@ref) for the description of keyword parameters).

_References_

[1] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.   
"""
function pfdric(A::PM1, C::PM2, R::PM3, Q::PM4, S::Union{PM5,Missing} = missing; kwargs...) where 
  {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  Xt, EVALS, Ft =  prdric(reverse(A)', reverse(C)', reverse(R), reverse(Q), ismissing(S) ? S : reverse(S); kwargs...) 
  return reverse(pmshift(Xt)), EVALS, reverse(Ft)'
end
function pfdric(A::PM1, C::PM2, R::AbstractMatrix, Q::PM4, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  nperiod = rationalize(A.period/A.Ts).num
  pfdric(A, C, PeriodicMatrix(R, A.period; nperiod), Q, ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicMatrix(S, A.period; nperiod) : S; kwargs...)
end
function pfdric(A::PM1, C::PM2, R::Union{PM4,AbstractMatrix}, Q::AbstractMatrix, S::Union{PM5,AbstractMatrix,Missing} = missing; kwargs...) where 
                {PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM4 <: PeriodicMatrix, PM5 <: PeriodicMatrix}
  ma, na = size(A)
  if maximum(na) == minimum(na) && maximum(ma) == minimum(ma)
     nperiod = rationalize(A.period/A.Ts).num
     pfdric(A, C, isa(R,AbstractMatrix) ? PeriodicMatrix(R, A.period; nperiod) : R, PeriodicMatrix(Q, A.period; nperiod), ismissing(S) ? S : isa(S,AbstractMatrix) ? PeriodicMatrix(S, A.period; nperiod) : S; kwargs...)
  else
     error("Constant Q and S are not possible for time-varying dimensions")
  end
end


