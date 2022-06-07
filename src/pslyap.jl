"""
    prlyap(A, C) -> X

Solve the reverse-time periodic discrete-time Lyapunov equation

    A'σXA + C = X

where `σ` is the forward shift operator `σX(i) = X(i+1)`.                 

For the periodic matrices `A` and `C` with the same sampling period and commensurate periods, 
the periodic matrix `X` is determined, whose period is automatically set to the
the least common period of `A` and `C`.  
"""
function prlyap(A::PeriodicArray, C::PeriodicArray) 
   A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
   period = promote_period(A, C)
   nta = numerator(rationalize(period/A.period))
   K = nta*A.nperiod*A.dperiod
   X = pslyapd(A.M, C.M; adj = true)
   return PeriodicArray(X, period; nperiod = div(K,size(X,3)))
end
prlyap(A::PeriodicArray, C::AbstractMatrix) = prlyap(A, PeriodicArray(C, A.Ts; nperiod = 1))
"""
    pflyap(A, C) -> X

Solve the forward-time periodic discrete-time Lyapunov equation

    A'XA + C = σX

where `σ` is the forward shift operator `σX(i) = X(i+1)`.                 

For the periodic matrices `A` and `C` with the same sampling period and commensurate periods, 
the periodic matrix `X` is determined, whose period is automatically set to the
the least common period of `A` and `C`.  
"""
function pflyap(A::PeriodicArray, C::PeriodicArray) 
   A.Ts ≈ C.Ts || error("A and C must have the same sampling time")
   period = promote_period(A, C)
   nta = numerator(rationalize(period/A.period))
   K = nta*A.nperiod*A.dperiod
   X = pslyapd(A.M, C.M; adj = false)
   return PeriodicArray(X, period; nperiod = div(K,size(X,3)))
end
pflyap(A::PeriodicArray, C::AbstractMatrix) = pflyap(A, PeriodicArray(C, A.Ts; nperiod = 1))
"""
    pslyapd(A, C; adj = true) -> X

Solve the periodic discrete-time Lyapunov equation.

For the square `n`-th order periodic matrices `A(i)`, `i = 1, ..., pa` and 
`C(i)`, `i = 1, ..., pc`  of periods `pa` and `pc`, respectively, 
the periodic solution `X(i)`, `i = 1, ..., p` of period `p = lcm(pa,pc)` of the 
periodic Lyapunov equation is solved:  for `adj = true`,  

    A(i)'*X(i+1)*A(i) + C(i) = X(i), i = 1, ..., p    

and for `adj = false`   

    A(i)*X(i)*A(i)' + C(i) = X(i+1), i = 1, ..., p.   

The periodic matrices `A` and `C` are stored in the `n×n×pa` and `n×n×pc` 3-dimensional 
arrays `A` and `C`, respectively, and `X` results as a `n×n×p` 3-dimensional array.  

The periodic discrete analog of the Bartels-Steward method based on the periodic Schur form
of the periodic matrix `A` is employed [1].

_Reference:_

[1] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
function pslyapd(A::AbstractArray{T1, 3}, C::AbstractArray{T2, 3}; adj::Bool = true) where {T1, T2}
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
   AS, Q, _, KSCHUR = pschur(A1)
   
   #X = Q'*C*Q
   X = Array{T,3}(undef, n, n, p)

   for i = 1:p
       ia = mod(i-1,pa)+1
       ic = mod(i-1,pc)+1
       ia1 = mod(i,pa)+1

       X[:,:,i] = adj ? utqu(view(C1,:,:,ic),view(Q,:,:,ia)) : 
                        utqu(view(C1,:,:,ic),view(Q,:,:,ia1)) 
   end
   # solve A'σXA - X + C = 0
   pdlyaps!(KSCHUR, AS, X; adj)

   #X <- Q*X*Q'
   for i = 1:p
       ia = mod(i-1,pa)+1
       utqu!(view(X,:,:,i),view(Q,:,:,ia)')
   end
   return X
end
"""
     pslyapdkr(A, C; adj = true) -> X

Solve the periodic discrete-time Lyapunov matrix equation

      A'σXA + C = X, if adj = true,

or 

      A*X*A' + C =  σX, if adj = false, 

where `σ` is the forward shift operator `σX(i) = X(i+1)`.                 
The Kronecker product expansion of equations is employed. `A` and `C` are
periodic square matrices, and `A` must not have characteristic multipliers on the unit circle.
This function is not recommended for large order matrices or large periods.
"""
function pslyapdkr(A::AbstractArray{T, 3}, C::AbstractArray{T, 3}; adj = true) where {T}
    m, n, pc = size(C)
    n == LinearAlgebra.checksquare(A[:,:,1]) 
    m == LinearAlgebra.checksquare(C[:,:,1]) 
    m == n  || throw(DimensionMismatch("A and C have incompatible dimensions"))
    pa = size(A,3)
    n2 = n*n
    p = lcm(pa,pc)
    N = p*n2
    R = zeros(T, N, N)
    adj ? copyto!(view(R,N-n2+1:N,1:n2),kron(A[:,:,pa]',A[:,:,pa]')) : 
          copyto!(view(R,1:n2,N-n2+1:N),kron(A[:,:,pa],A[:,:,pa])) 
    (i2, j2) = adj ? (n2, n2+n2) : (n2+n2, n2)
    for i = 1:p-1
        i1 = i2-n2+1
        j1 = j2-n2+1
        ia = mod(i-1,pa)+1
        adj ? copyto!(view(R,i1:i2,j1:j2),kron(A[:,:,ia]',A[:,:,ia]')) : 
              copyto!(view(R,i1:i2,j1:j2),kron(A[:,:,ia],A[:,:,ia])) 
        i2 += n2
        j2 += n2
    end
    indc = mod.(0:p-1,pc).+1
    return adj ? reshape((I-R) \ (C[:,:,indc][:]), n, n, p) : 
    reshape((I-R) \ (C[:,:,circshift(indc,1)][:]), n, n, p)
end
function pdlyaps!(KSCHUR::Int, A::StridedArray{T1,3}, C::StridedArray{T1,3}; adj = true) where {T1<:BlasReal}
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
function dpsylv2(REV::Bool, N1::Int, N2::Int, KSCHUR::Int, TL::StridedArray{T,3}, TR::StridedArray{T,3}, 
                 B::StridedArray{T,3}, W::AbstractMatrix{T}, WX::AbstractMatrix{T}) where {T}
#     To solve for the N1-by-N2 matrices X_j, j = 1, ..., P, 
#     1 <= N1,N2 <= 2, in the P simultaneous equations: 

#     if REV = true

#       TL_j'*X_(j+1)*TR_j - X_j = B_j, X_(P+1) = X_1  (1) 

#     or if REV = false

#       TL_j*X_j*TR_j' - X_(j+1) = B_j, X_(P+1) = X_1  (2)

#     where TL_j is N1 by N1, TR_j is N2 by N2, B_j is N1 by N2,
#     and ISGN = 1 or -1.  

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

   ZOLD[i1,i2] = Z[i1,i2]

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
                 luslv!(ATMP,view(BTMP,1:2,1:1))
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
                 luslv!(ATMP,view(Z,1:2,1:1))
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
                 luslv!(ATMP,view(BTMP,1:2,1:2))
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
                 luslv!(ATMP,view(Z,1:2,1:2))
                 #Z = ATMP\Z
                 X[ 1, 1, J1 ] = Z[ 1, 1 ]
                 X[ 1, 2, J1 ] = Z[ 1, 2 ]
                 X[ 2, 1, J1 ] = Z[ 2, 1 ]
                 X[ 2, 2, J1 ] = Z[ 2, 2 ]
              end
          end
       end
       DNORM = zero(T)
       for J = 1:N2
          for I = 1:N1
             DNORM = max( DNORM, abs( ZOLD[I,J] - Z[I,J] ) )
             ZOLD[ I, J ] = Z[ I, J ]
          end
       end
       #println("XNORM = $XNORM DNORM = $DNORM")
       DNORM <= EPSM*XNORM && (return -X)
   end
   @warn "iterative process not converging: solution may be inaccurate"
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
