"""
     phess(A::Array{Float64,3}; hind = 1, rev = true, withZ = true) -> (H, Z, ihess)
     phess1(A::Array{Float64,3}; hind = 1, rev = true, withZ = true) -> (H, Z, ihess)

Compute the Hessenberg decomposition of a product of square matrices 
`A(p)*...*A(2)*A(1)`, if `rev = true` (default) or `A(1)*A(2)*...*A(p)`
if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `n×n×p` array `A` 
such that the `i`-th matrix `A(i)` is contained in `A[:,:,i]`.
The resulting `n×n×p` arrays `H` and `Z` contain the matrices `H(1)`, `...`, `H(p)`
and the orthogonal matrices `Z(1)`, `...`, `Z(p)`, respectively, 
such that for `rev = true`

           Z(2)' * A(1) * Z(1) = H(1),
           Z(3)' * A(2) * Z(2) = H(2),
                  ...
           Z(1)' * A(p) * Z(p) = H(p),

and for `rev = false`

           Z(1)' * A(1) * Z(2) = H(1),
           Z(2)' * A(2) * Z(3) = H(2),
                  ...
           Z(p)' * A(p) * Z(1) = H(p).

If `hind = ihess`, with `1 ≤ ihess ≤ p` (default `ihess = 1`), then 
`H(i)`, `i = 1, ..., p` are in a periodic Hessenberg form, 
with `H(ihess)` in upper Hessenberg form and `H(i)` 
upper triangular for `i` ``\\neq`` `ihess`. 
`H(i)` and `Z(i)` are contained in `H[:,:,i]` and `Z[:,:,i]`, respectively. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 

The function `phess` is based on a wrapper for the SLICOT subroutine `MB03VW`
   (see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref)).
   
The function `phess1` is based on wrappers for the SLICOT subroutines `MB03VD`  
(see [`PeriodicSystems.SLICOTtools.mb03vd!`](@ref)) and `MB03VY`  (see [`PeriodicSystems.SLICOTtools.mb03vy!`](@ref)).
"""
function phess(A::AbstractArray{Float64,3}; kwargs...)
   phess!(copy(A), (1,size(A,1)); kwargs...)
end

"""

     phess!(A::Array{Float64,3}, ilh::Tuple(Int,Int) = (1, size(A,1)); kwargs...) -> (H, Z, ihess)

Same as `phess(A; kwargs...)` but uses the input matrix `A` as workspace and specifies a range `ilh = (ilo, ihi)`, such that
all matrices `A(j), j = 1, ..., p`, are already in periodic Hessenberg forms in rows and columns `1:ilo-1` and `ihi+1:n`, where
`n` is the first dimension of `A`.
"""
function phess!(A::AbstractArray{Float64,3}, ilh::Tuple{Int,Int} = (1, size(A,1)); rev::Bool = true, hind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions")
   p = size(A,3)
   (hind < 1 || hind > p) && error("hind is out of range $(1:p)") 
   (ilo, ihi) = ilh
   (ilo < 1 || ihi < ilo || ihi > n) && error("ilo and ihi must satisfy 1 ≤ ilo ≤ ihi ≤ p") 

   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'I'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end
   
   hc = rev ? p - hind + 1 : hind
   rev && (reverse!(A, dims = 3))

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   
   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, ones(BlasInt,p), A, Z, LIWORK, LDWORK)

   if rev
      return reverse!(A, dims = 3), withZ ? Z[:,:,mod.(p:-1:1,p).+1] : nothing, hind
   else
      return A, withZ ? Z : nothing, hind
   end
end
function phess1(A::AbstractArray{Float64,3}; rev::Bool = true, hind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions")
   K = size(A,3)
   (hind < 1 || hind > K) && error("hind is out of range $(1:K)") 

   shift = (hind != 1)
   ilo = 1; ihi = n; 
   withZ && (Q = Array{Float64,3}(undef, n, n, K))
   TAU = Array{Float64,2}(undef, max(n-1,1), K)
   
   if shift 
     if rev 
        imap = mod.(Vector(hind+K-1:-1:hind),K) .+ 1
        H = A[:,:,imap]
    else
        # shift h, h+1, ..., h+K-1 matrices to position 1, 2, ..., K
        imap = mod.(Vector(hind-1:hind+K-2),K) .+ 1
        #rev && (ind = reverse(ind); h = K-hind+1 )
        H = copy(view(A,:,:,imap))
     end
   else
      if rev 
         imap = mod.(Vector(K:-1:1),K) .+ 1
         H = A[:,:,imap]
      else
         H = copy(A); imap = 1:K; 
      end
   end

   # reduce to periodic Hessenberg form
   SLICOTtools.mb03vd!(n, K, ilo, ihi,  H, TAU)
   for i in 1:K
       withZ && (Q[:,:,i] .= tril(view(H,:,:,i)))    
       if n > 1 
          if ( n>2 && i == 1 )
             triu!(view(H,:,:,i),-1)
   
           elseif ( i>1 )
              triu!(view(H,:,:,i))
           end 
       end 
   end 
   
   # accumulate transformations
   withZ && SLICOTtools.mb03vy!(n, K, ilo, ihi, Q, TAU)

   if shift
      if rev
         return H[:,:,imap], withZ ? Q[:,:,mod.(K+hind:-1:hind+1,K).+1] : nothing, hind
      else
         # return back shifted matrices
         #imap1 = mod.(imap.+(hind+1),K).+1
         imap1 = invperm(imap)
         return H[:,:,imap1], withZ ? Q[:,:,imap1] : nothing, hind
      end
   else
     if rev 
        return H[:,:,imap], withZ ? Q[:,:,mod.(K+1:-1:2,K).+1] : nothing, hind
     else
        return H, withZ ? Q : nothing, hind
     end
  end
end
"""
     pschur(A::Array{Float64,3}; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur1(A::Array{Float64,3}; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur2(A::Array{Float64,3}; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

Compute the Schur decomposition of a product of square matrices 
`A(p)*...*A(2)*A(1)`, if `rev = true` (default) or `A(1)*A(2)*...*A(p)`
if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `n×n×p` array `A` 
such that the `i`-th matrix `A(i)` is contained in `A[:,:,i]`.
The resulting `n×n×p` arrays `S` and `Z` contain the matrices `S(1)`, `...`, `S(p)`
and the orthogonal matrices `Z(1)`, `...`, `Z(p)`, respectively, 
such that for `rev = true`

           Z(2)' * A(1) * Z(1) = S(1),
           Z(3)' * A(2) * Z(2) = S(2),
                  ...
           Z(1)' * A(p) * Z(p) = S(p),

and for `rev = false`

           Z(1)' * A(1) * Z(2) = S(1),
           Z(2)' * A(2) * Z(3) = S(2),
                  ...
           Z(p)' * A(p) * Z(1) = S(p).

If `sind = ischur`, with `1 ≤ ischur ≤ p` (default `ischur = 1`), then 
`S(i)`, for `i = 1, ..., p` are in a periodic Schur form, 
with `S(ischur)` in quasi-upper triangular (or Schur) form and `S(i)` 
upper triangular for `i` ``\\neq`` `ischur`. 
`S(i)` and `Z(i)` are contained in `S[:,:,i]` and `Z[:,:,i]`, respectively. 
The vector `ev` contains the eigenvalues of the appropriate matrix product. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 

The function `pschur` is based on wrappers for the SLICOT subroutines `MB03VW`  
(see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref)) and `MB03BD`  (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
The function `pschur1` is based on wrappers for the SLICOT subroutines `MB03VD`  (see [`PeriodicSystems.SLICOTtools.mb03vd!`](@ref)), 
`MB03VY`  (see [`PeriodicSystems.SLICOTtools.mb03vy!`](@ref))  and `MB03BD`  (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].

The function `pschur2` is based on wrappers for the SLICOT subroutines `MB03VD`  (see [`PeriodicSystems.SLICOTtools.mb03vd!`](@ref)), 
`MB03VY`  (see [`PeriodicSystems.SLICOTtools.mb03vy!`](@ref)) and `MB03VW`  (see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref)), 
based on the algorithm proposed in [1]. Known issue: `MB03VW` may fails for larger periods. 

REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.
"""
function pschur(A::AbstractArray{Float64,3}; kwargs...)
   pschur!(copy(A), (1,size(A,1)); kwargs...)
end
"""

     pschur!(A::Array{Float64,3}, ilh::Tuple(Int,Int) = (1, size(A,1)); kwargs...) -> (S, Z, ihess)

Same as `pschur(A; kwargs...)` but uses the input matrix `A` as workspace and specifies a range `ilh = (ilo, ihi)`, such that
all matrices `A(j), j = 1, ..., p`, are already in periodic Schur forms in rows and columns `1:ilo-1` and `ihi+1:n`, where
`n` is the first dimension of `A`.
"""
function pschur!(A::AbstractArray{Float64,3}, ilh::Tuple{Int,Int} = (1, size(A,1)); rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   (ilo, ihi) = ilh 
   
   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'I'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   hc = rev ? p - sind + 1 : sind
   rev && (reverse!(A, dims = 3))

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   SIND = ones(BlasInt,p)

   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, A, Z, LIWORK, LDWORK)

   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, hc, ilo, ihi, SIND, A, Z, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ

   if rev
      return reverse!(A, dims = 3), withZ ? Z[:,:,mod.(p:-1:1,p).+1] : nothing, ev, sind, α, γ
   else
      return A, withZ ? Z : nothing, ev, sind, α, γ
   end
end
function pschur!(A::AbstractArray{Float64,3}, Z::AbstractArray{Float64,3}, ilh::Tuple{Int,Int} = (1, size(A,1)); rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   (ilo, ihi) = ilh 
   
   compQ = withZ ? 'I' : 'N'
   
   hc = rev ? p - sind + 1 : sind
   rev && (reverse!(A, dims = 3))

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   SIND = ones(BlasInt,p)

   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, A, Z, LIWORK, LDWORK)

   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, hc, ilo, ihi, SIND, A, Z, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)

   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ

   if rev 
      reverse!(A, dims = 3)
      withZ && reverse!(view(Z,:,:,2:p),dims=3)
   end
   return ev, sind, α, γ
end
function pschur!(ws_pschur::Tuple, A::AbstractArray{Float64,3}, Z::AbstractArray{Float64,3}, ilh::Tuple{Int,Int} = (1, size(A,1)); rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   (ilo, ihi) = ilh 
   
   compQ = withZ ? 'I' : 'N'

   hc = rev ? p - sind + 1 : sind
   rev && (reverse!(A, dims = 3))

   (QIND, SIND, ALPHAR, ALPHAI, BETA, SCAL, IWORK, DWORK) = ws_pschur

   # reduce to periodic Hessenberg form
   fill!(SIND, one(BlasInt))
   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, A, Z, IWORK, DWORK)

   # reduce to periodic Schur form
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, hc, ilo, ihi, SIND, A, Z, ALPHAR, ALPHAI, BETA, SCAL, IWORK, DWORK)

   # α = complex.(ALPHAR, ALPHAI) ./ BETA
   # γ = 2. .^SCAL
   # ev = α .* γ
   ev = complex.(ALPHAR, ALPHAI)
   ev .*= (2. .^SCAL ./ BETA)

   if rev 
      reverse!(A, dims = 3)
      withZ && reverse!(view(Z,:,:,2:p),dims=3)
   end
   return ev, sind
end
function ws_pschur(n, p)
   LIWORK = max(3*p, 2*p + n, 1)
   IWORK = Array{BlasInt,1}(undef, LIWORK)
   LDWORK = max(1, p + max( 2*n, 8*p ))
   DWORK = Array{Float64,1}(undef, LDWORK)
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   SIND = Array{BlasInt,1}(undef, p)
   QIND = Array{BlasInt,1}(undef, 1) 
   return (QIND, SIND, ALPHAR, ALPHAI, BETA, SCAL, IWORK, DWORK)
end
function pschur1(A::Array{Float64,3}; rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   shift = (sind != 1)
   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n; 
   TAU = Array{Float64,2}(undef, max(n-1,1), p)
   
   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'U'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   if shift 
     if rev 
        imap = mod.(Vector(sind+p-1:-1:sind),p) .+ 1
        S = A[:,:,imap]
    else
        # shift h, h+1, ..., h+p-1 matrices to position 1, 2, ..., p
        imap = mod.(Vector(sind-1:sind+p-2),p) .+ 1
        #rev && (ind = reverse(ind); h = p-hind+1 )
        S = copy(view(A,:,:,imap))
     end
   else
      if rev 
         imap = mod.(Vector(p:-1:1),p) .+ 1
         S = A[:,:,imap]
      else
         S = copy(A); imap = 1:p; 
      end
   end

   # reduce to periodic Hessenberg form
   SLICOTtools.mb03vd!(n, p, ilo, ihi,  S, TAU)
   for i in 1:p
       withZ && (Z[:,:,i] .= tril(view(S,:,:,i)))    
       if n > 1 
          if n > 2 && i == 1 
             triu!(view(S,:,:,i),-1)  
          elseif i > 1 
             triu!(view(S,:,:,i))
          end 
       end 
   end 
   
   # accumulate transformations
   withZ && SLICOTtools.mb03vy!(n, p, ilo, ihi, Z, TAU)


   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   QIND = Array{BlasInt,1}(undef, 1) 
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, 1, ilo, ihi, ones(BlasInt,p), S, Z, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ

   if shift
     if rev
        return S[:,:,imap], withZ ? Z[:,:,mod.(p+sind:-1:sind+1,p).+1] : nothing, ev, sind, α, γ
     else
        # return back shifted matrices
        imap1 = invperm(imap)
        return S[:,:,imap1], withZ ? Z[:,:,imap1] : nothing, ev, sind, α, γ
     end
   else
     if rev 
        return S[:,:,imap], withZ ? Z[:,:,mod.(p+1:-1:2,p).+1] : nothing, ev, sind, α, γ
     else
        return S, withZ ? Z : nothing, ev, sind, α, γ
     end
  end
end
function pschur2(A::Array{Float64,3}; rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   shift = (sind != 1)
   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n; 
   TAU = Array{Float64,2}(undef, max(n-1,1), p)
   
   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'V'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   if shift 
     if rev 
        imap = mod.(Vector(sind+p-1:-1:sind),p) .+ 1
        S = A[:,:,imap]
    else
        # shift h, h+1, ..., h+p-1 matrices to position 1, 2, ..., p
        imap = mod.(Vector(sind-1:sind+p-2),p) .+ 1
        #rev && (ind = reverse(ind); h = p-hind+1 )
        S = copy(view(A,:,:,imap))
     end
   else
      if rev 
         imap = mod.(Vector(p:-1:1),p) .+ 1
         S = A[:,:,imap]
      else
         S = copy(A); imap = 1:p; 
      end
   end

   # reduce to periodic Hessenberg form
   SLICOTtools.mb03vd!(n, p, ilo, ihi,  S, TAU)
   for i in 1:p
       withZ && (Z[:,:,i] .= tril(view(S,:,:,i)))    
       if n > 1 
          if n > 2 && i == 1 
             triu!(view(S,:,:,i),-1)  
          elseif i > 1 
             triu!(view(S,:,:,i))
          end 
       end 
   end 
   
   # accumulate transformations
   withZ && SLICOTtools.mb03vy!(n, p, ilo, ihi, Z, TAU)


   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   QIND = Array{BlasInt,1}(undef, 1) 
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   SLICOTtools.mb03wd!('S',compQ, n, p, ilo, ihi, ilo, ihi, S, Z, ALPHAR, ALPHAI, LDWORK)
   ev = complex.(ALPHAR, ALPHAI)

   if shift
     if rev
        return S[:,:,imap], withZ ? Z[:,:,mod.(p+sind:-1:sind+1,p).+1] : nothing, ev, sind
     else
        # return back shifted matrices
        imap1 = invperm(imap)
        return S[:,:,imap1], withZ ? Z[:,:,imap1] : nothing, ev, sind
     end
   else
     if rev 
        return S[:,:,imap], withZ ? Z[:,:,mod.(p+1:-1:2,p).+1] : nothing, ev, sind
     else
        return S, withZ ? Z : nothing, ev, sind
     end
  end
end

"""
     pschur(A::Vector{Matrix{T}}; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur1(A::Vector{Matrix{T}}; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

Compute the extended periodic Schur decomposition of a square product of matrices 
`A(p)*...*A(2)*A(1)`, if `rev = true` (default) or `A(1)*A(2)*...*A(p)`
if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `p`-vector of matrices `A` 
such that the `i`-th matrix  `A(i)`, of dimensions `m(i)×n(i)`, is contained in `A[i]`.
The resulting `p`-vectors `S` and `Z` contain the matrices `S(1)`, `...`, `S(p)`
and the orthogonal matrices `Z(1)`, `...`, `Z(p)`, respectively, 
such that for `rev = true`

           Z(2)' * A(1) * Z(1) = S(1),
           Z(3)' * A(2) * Z(2) = S(2),
                  ...
           Z(1)' * A(p) * Z(p) = S(p),

and for `rev = false`

           Z(1)' * A(1) * Z(2) = S(1),
           Z(2)' * A(2) * Z(3) = S(2),
                  ...
           Z(p)' * A(p) * Z(1) = S(p).

The resulting index `ischur` is determined such that `m(ischur) ≤ m(i), ∀i`.
The resulting `S(i)`, for `i = 1, ..., p` are in an extended  periodic Schur form, 
with `S(ischur)` in a quasi-upper trapezoidal form and `S(i)` 
upper trapezoidal for `i` ``\\neq`` `ischur`. 
`S(i)` and `Z(i)` are contained in `S[i]` and `Z[i]`, respectively. 
The first `nmin` components of `ev := α .* γ` contain the _core eigenvalues_ of the appropriate matrix product,
where `nmin = m(ischur)`, while the last `nmax-nmin` components of `ev` are zero, 
where `nmax` is the largest row or column dimension of `A(i)`, for `i = 1, ..., p`. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 

The function `pschur` is based on wrappers for the SLICOT subroutines `MB03VW` (see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref))
 and `MB03BD` (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
The function `pschur1` is based on wrappers for the SLICOT subroutines `MB03VD` (see [`PeriodicSystems.SLICOTtools.mb03vd!`](@ref)), 
`MB03VY` (see [`PeriodicSystems.SLICOTtools.mb03vy!`](@ref)),  and `MB03BD` (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].

REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.

"""
function pschur(A::AbstractVector{Matrix{T}}; rev::Bool = true, withZ::Bool = true) where {T<:Real}

   p = length(A) 
   mp, np = size.(A,1), size.(A,2) 
   if rev
      all(mp .== view(np,mod.(1:p,p).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   else
      all(np .== view(mp,mod.(1:p,p).+1)) ||  
         error("the number of columns of A[i] must be equal to the number of rows of A[i+1]")
   end
   sind = argmin(mp)
   nmin = mp[sind]
   n = maximum(np)

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n
   St = zeros(Float64, n, n, p)
   #TAUt = Array{Float64,2}(undef, max(n-1,1), p)
   if withZ 
      Zt = Array{Float64,3}(undef, n, n, p) 
      compQ = 'I'
   else
      Zt = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end
    
   if rev 
      imap = mod.(Vector(sind+p-1:-1:sind),p) .+ 1
      [(k = imap[i]; St[1:mp[k],1:np[k],i] = A[k];) for i in 1:p]
      hc = 1
   else
      # shift h, h+1, ..., h+p-1 matrices to position 1, 2, ..., p
      imap = mod.(Vector(sind-1:sind+p-2),p) .+ 1
      [(St[1:mp[i],1:np[i],i] = A[i]) for i in 1:p]
      hc = sind
   end

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   SIND = ones(BlasInt,p)

   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, St, Zt, LIWORK, LDWORK)
   
   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   # use ilo = 1 and ihi = nmin
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C', compQ, QIND, p, n, hc, ilo, nmin, SIND, St, Zt, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ

   if rev
      imap1 = mod.(p+sind:-1:sind+1,p).+1
      return [St[1:mp[i],1:np[i],imap[i]] for i in 1:p], 
             withZ ? [Zt[1:np[i],1:np[i],imap1[i]] for i in 1:p] : nothing, ev, sind, α, γ
   else
      # return back shifted matrices
      imap1 = mod.(imap.+(sind+1),p).+1
      return [St[1:mp[i],1:np[i],i] for i in 1:p], 
             withZ ? [Zt[1:mp[i],1:mp[i],i] for i in 1:p] : nothing, ev, sind, α, γ
   end
end
function pschur1(A::AbstractVector{Matrix{T}}; rev::Bool = true, withZ::Bool = true) where {T <: Real}

   p = length(A) 
   mp, np = size.(A,1), size.(A,2) 
   if rev
      all(mp .== view(np,mod.(1:p,p).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
   else
      all(np .== view(mp,mod.(1:p,p).+1)) ||  
         error("the number of columns of A[i] must be equal to the number of rows of A[i+1]")
   end
   sind = argmin(mp)
   nmin = mp[sind]
   n = maximum(np)

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n
   St = zeros(Float64, n, n, p)
   TAUt = Array{Float64,2}(undef, max(n-1,1), p)
   if withZ 
      Zt = Array{Float64,3}(undef, n, n, p) 
      compQ = 'U'
   else
      Zt = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end
    
   if rev 
      imap = mod.(Vector(sind+p-1:-1:sind),p) .+ 1
      [(k = imap[i]; St[1:mp[k],1:np[k],i] = A[k];) for i in 1:p]
   else
      # shift h, h+1, ..., h+p-1 matrices to position 1, 2, ..., p
      imap = mod.(Vector(sind-1:sind+p-2),p) .+ 1
      [(k = imap[i]; St[1:mp[k],1:np[k],i] = A[k]) for i in 1:p]
   end

   # reduce to periodic Hessenberg form
   SLICOTtools.mb03vd!(n, p, ilo, ihi,  St, TAUt)
   for i in 1:p
       withZ && (Zt[:,:,i] .= tril(view(St,:,:,i)) )   
       if n > 1 
          if n > 2 && i == 1 
             triu!(view(St,:,:,i),-1)  
          elseif i > 1 
             triu!(view(St,:,:,i))
          end 
       end 
   end 
   
   # accumulate transformations
   withZ && SLICOTtools.mb03vy!(n, p, ilo, ihi, Zt, TAUt)

   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   QIND = Array{BlasInt,1}(undef, 1) 
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   # use ilo = 1 and ihi = nmin
   SLICOTtools.mb03bd!('T','C', compQ, QIND, p, n, 1, ilo, nmin, ones(BlasInt,p), St, Zt, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ

   if rev
      imap1 = mod.(p+sind:-1:sind+1,p).+1
      return [St[1:mp[i],1:np[i],imap[i]] for i in 1:p], 
             withZ ? [Zt[1:np[i],1:np[i],imap1[i]] for i in 1:p] : nothing, ev, sind, α, γ
   else
      # return back shifted matrices
      imap1 = mod.(imap.+(sind+1),p).+1
      return [St[1:mp[i],1:np[i],imap1[i]] for i in 1:p], 
             withZ ? [Zt[1:mp[i],1:mp[i],imap1[i]] for i in 1:p] : nothing, ev, sind, α, γ
   end
end

"""
     pschur(A::AbstractArray{T,3}, E::AbstractArray{T,3}; rev = true, withZ = true) -> (S, T, Q, Z, ev, ischur, α, γ)

Compute the periodic Schur decomposition of a square formal quotient product of matrices 
`inv(E(p))*A(p)*...*inv(E(2))*A(2)*inv(E(1))*A(1)`, if `rev = true` (default) or 
`A(1)*inv(E(1))*A(2)*inv(E(2))*...*A(p)*inv(E(p))` if `rev = false`, without evaluating the product
and the inverses. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `n×n×p`-array `A` 
such that the `i`-th matrix  `A(i)` is contained in `A[:,:,i]`.
The square matrices `E(1)`, `...`, `E(p)` are contained in the `n×n×p`-array `E` 
such that the `i`-th matrix  `E(i)` is contained in `E[:,:,i]`.

The resulting `n×n×p`-arrays `S`, `T`, `Q` and `Z` contain, respectively, 
the matrices `S(1)`, `...`, `S(p)` with `S(ischur)` in a quasi-upper trapezoidal form and 
`S(i)` upper trapezoidal for `i` ``\\neq`` `ischur`,
the upper triangular matrices `T(1)`, `...`, `T(p)`, 
the orthogonal matrices `Q(1)`, `...`, `Q(p)`, and `Z(1)`, `...`, `Z(p)`, 
such that for `rev = true`

           Q(1)' * A(1) * Z(1) = S(1),  Q(1)' * E(1) * Z(2) = T(1), 
           Q(2)' * A(2) * Z(2) = S(2),  Q(2)' * E(2) * Z(3) = T(2),
                  ...
           Q(p)' * A(p) * Z(p) = S(p),  Q(p)' * E(p) * Z(1) = T(p),

and for `rev = false`

           Q(1)' * A(1) * Z(1) = S(1),  Q(2)' * E(1) * Z(1) = T(1), 
           Q(2)' * A(2) * Z(2) = S(2),  Q(3)' * E(2) * Z(2) = T(2),
                  ...
           Q(p)' * A(p) * Z(p) = S(p),  Q(1)' * E(p) * Z(p) = T(p),

The complex vector `ev` contains the eigenvalues of the appropriate matrix product,
and can be alternatively expressed as `ev := α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case  `Q = nothing` and `Z = nothing`. 

The function `pschur` is based on wrappers for the SLICOT subroutines `MB03VW` (see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref))
 and `MB03BD` (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.

"""
function pschur(A::AbstractArray{T,3}, E::AbstractArray{T,3}; rev::Bool = true, withZ::Bool = true) where {T <: Real}

   p = size(A,3) 
   p == size(E,3) || error("A and E must have the same third dimension")
   n = size(A,2)
   size(A,1) == n || throw(DimensionMismatch("A must have equal first ans second dimensions"))
   (size(E,1) == n && size(E,2) == n) || throw(DimensionMismatch("E must have the same first and second dimensions as A"))
   
   p2 = 2*p
   St = Array{Float64,3}(undef,n,n,p2)
   [(copyto!(view(St,:,:,2*i-1),A[:,:,i]); copyto!(view(St,:,:,2*i),E[:,:,i])) for i in 1:p ]
   S = trues(p2); [S[2*i] = false for i in 1:p]
   St, Zt, ev, sind, α, γ = pgschur(St, S; rev, withZ)
   
   return St[:,:,1:2:p2], St[:,:,2:2:p2], withZ ? Zt[:,:,(rev ? 2 : 1):2:p2] : nothing, withZ ? Zt[:,:,(rev ? 1 : 2):2:p2] : nothing, ev, sind, α, γ 
end

 
"""
     pschur(A::Vector{Matrix{T}}, E::Vector{Matrix{T}}; rev = true, withZ = true) -> (S, T, Q, Z, ev, ischur, α, γ)

Compute the extended periodic Schur decomposition of a square formal product of matrices 
`inv(E(p))*A(p)*...*inv(E(2))*A(2)*inv(E(1))*A(1)`, if `rev = true` (default) or 
`A(1)*inv(E(1))*A(2)*inv(E(2))*...*A(p)*inv(E(p))` if `rev = false`, without evaluating the product
and the inverses. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `p`-vector of matrices `A` 
such that the `i`-th matrix  `A(i)`, of dimensions `m(i)×n(i)`, is contained in `A[i]`.
The square matrices `E(1)`, `...`, `E(p)` are contained in the `p`-vector of matrices `E` 
such that the `i`-th matrix  `E(i)`, of dimensions `m(i)×m(i)` if `rev = true` or `n(i)×n(i)` if `rev = false`, 
is contained in `E[i]`.

The resulting index `ischur` is determined such that `m(ischur) ≤ m(i), ∀i`.
The resulting `p`-vectors `S`, `T`, `Q` and `Z` contain, respectively, 
the matrices `S(1)`, `...`, `S(p)` with `S(ischur)` in a quasi-upper trapezoidal form and 
`S(i)` upper trapezoidal for `i` ``\\neq`` `ischur`,
the upper triangular matrices `T(1)`, `...`, `T(p)`, 
the orthogonal matrices `Q(1)`, `...`, `Q(p)`, and `Z(1)`, `...`, `Z(p)`, 
such that for `rev = true`

           Q(1)' * A(1) * Z(1) = S(1),  Q(1)' * E(1) * Z(2) = T(1), 
           Q(2)' * A(2) * Z(2) = S(2),  Q(2)' * E(2) * Z(3) = T(2),
                  ...
           Q(p)' * A(p) * Z(p) = S(p),  Q(p)' * E(p) * Z(1) = T(p),

and for `rev = false`

           Q(1)' * A(1) * Z(1) = S(1),  Q(2)' * E(1) * Z(1) = T(1), 
           Q(2)' * A(2) * Z(2) = S(2),  Q(3)' * E(2) * Z(2) = T(2),
                  ...
           Q(p)' * A(p) * Z(p) = S(p),  Q(1)' * E(p) * Z(p) = T(p),

The first `nmin` components of `ev := α .* γ` contain the _core eigenvalues_ of the appropriate matrix product,
where `nmin = m(ischur)`, while the last `nmax-nmin` components of `ev` are zero, 
where `nmax` is the largest row or column dimension of `A(i)`, for `i = 1, ..., p`. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case  `Q = nothing` and `Z = nothing`. 

The function `pschur` is based on wrappers for the SLICOT subroutines `MB03VW` (see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref))
 and `MB03BD` (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.

"""
function pschur(A::AbstractVector{Matrix{T}}, E::AbstractVector{Matrix{T}}; rev::Bool = true, withZ::Bool = true) where {T <: Real}

   p = length(A) 
   p == length(E) || error("A and E must have the same length")
   mp, np = size.(A,1), size.(A,2) 
   mp1, np1 = size.(E,1), size.(E,2) 
   all(mp1 .== np1) || error("E must contain only square matrices")
   if rev
      all(mp .== view(np,mod.(1:p,p).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
      all(mp .== mp1) || error("the number of rows of A[i] must be equal to the order of E[i]")
   else
      all(np .== view(mp,mod.(1:p,p).+1)) ||  
         error("the number of columns of A[i] must be equal to the number of rows of A[i+1]")
      all(np .== np1) || error("the number of columns of A[i] must be equal to the order of E[i]")
   end
   
   p2 = 2*p
   St = Matrix{T}[]; [push!(push!(St,A[i]),E[i]) for i in 1:p]
   S = trues(p2); [S[2*i] = false for i in 1:p]
   St, Zt, ev, sind, α, γ = pgschur(St, S; rev, withZ)
   
   return St[1:2:p2], St[2:2:p2], withZ ? Zt[(rev ? 2 : 1):2:p2] : nothing, withZ ? Zt[(rev ? 1 : 2):2:p2] : nothing, ev, sind, α, γ 
end
"""
     pgschur(A::Vector{Matrix}, s::Union{Vector{Bool},BitVector}; rev = true, withQ = true) -> (S, Z, ev, ischur, α, γ)

Compute the generalized real periodic Schur decomposition of a formal product of square matrices 
`A(p)^s(p)*...A(2)^s(2)*A(1)^s(1)`, if `rev = true` (default), or 
`A(1)^s(1)*A(2)^s(2)*...*A(p)^s(p)`, if `rev = false`, where 's(j) = ±1'. 
The matrices `A(1)`, `A(2)`, `...`, `A(p)` are contained in the `p`-dimensional array `A` 
such that the `i`-th matrix  `A(i)` is contained in `A[i]`. 

The resulting `p`-dimensional array `S` contains the matrices `S(1)`, `...`, `S(p)` 
such that the `i`-th matrix `S(i)` is contained in `S[i]`. 
The component matrix `S[ischur]` is in a quasi-upper triangular form, while
`S[i]` is upper triangular for `i` ``\\neq`` `ischur`. 
If `withZ = true` (default), the resulting `p`-dimensional array `Z` contains the orthogonal transformation 
matrices `Z(1)`, `...`, `Z(p)` such that the `i`-th matrix `Z(i)` is contained in `Z[i]`. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 

The resulting matrices satisfy for `rev = true`

           Z(mod(j,p)+1)' * A(j) * Z(j) = S(j),  if S[j] = true, 
           Z(j)' * A(j) * Z(mod(j,p)+1) = S(j),  if S[j] = false, 

and for `rev = false`

           Z(j)' * A(j) * Z(mod(j,p)+1) = S(j),  if S[j] = true, 
           Z(mod(j,p)+1)' * A(j) * Z(j) = S(j),  if S[j] = false.

The vector `ev` contains the eigenvalues of the appropriate matrix product. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 

The function `pgschur` is based on wrappers for the SLICOT subroutines `MB03VW`  
(see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref)) and `MB03BD`  (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.
"""
function pgschur(A::AbstractVector{Matrix{T}}, S::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, withZ::Bool = true) where {T <: Real}
   p = length(A)
   p == length(S) || throw(DimensionMismatch("length of S must be equal to the length of A"))
   mp, np = size.(A,1), size.(A,2) 
   all(mp[.!S] .== np[.!S]) || throw(DimensionMismatch("only square matrices supported for S(i) = false"))
   n = maximum(np)
   if rev
      all(mp .== view(np,mod.(1:p,p).+1)) || 
         throw(DimensionMismatch("the number of columns of A[i+1] must be equal to the number of rows of A[i]"))
   else
      all(np .== view(mp,mod.(1:p,p).+1)) ||  
         throw(DimensionMismatch("the number of columns of A[i] must be equal to the number of rows of A[i+1]"))
   end
   if any(S)
      schurindex = rev ? argmin(mp) : mod(argmin(np),p)+1
      while !S[schurindex]
         schurindex = mod(schurindex,p)+1
      end
   else
      schurindex = 1
   end  
   ZERA = zeros(n,n)
   EYE = eye(Float64,n,n)
   St = Array{Float64,3}(undef,n, n, p)
   [(S[i] ? copyto!(view(St,:,:,i), ZERA) : copyto!(view(St,:,:,i), EYE) ) for i in 1:p]
   [copyto!(view(St,1:mp[i],1:np[i],i), A[i]) for i in 1:p]
   St, Zt, ev, ischur, α, γ = pgschur!(St, S; rev, withZ, schurindex)
   if rev
      return [St[1:mp[i],1:np[i],i] for i in 1:p],
             withZ ? [(S[i] ? Zt[1:np[i],1:np[i],i] : Zt[1:np[mod(i-1,p)+1],1:np[mod(i-1,p)+1],i]) for i in 1:p] : nothing, 
             ev, ischur, α, γ
   else
      return [St[1:mp[i],1:np[i],i] for i in 1:p], 
             withZ ? [(S[i] ? Zt[1:mp[mod(i-1,p)+1],1:mp[mod(i-1,p)+1],i] : Zt[1:np[i],1:np[i],i]) for i in 1:p] : nothing, 
             ev, ischur, α, γ
   end
end
"""
     pgschur(A::Array{Float64,3}, s::Union{Vector{Bool},BitVector}; rev = true, withQ = true) -> (S, Z, ev, ischur, α, γ)

Compute the generalized real periodic Schur decomposition of a formal product of square matrices 
`A(p)^s(p)*...A(2)^s(2)*A(1)^s(1)`, if `rev = true` (default), or 
`A(1)^s(1)*A(2)^s(2)*...*A(p)^s(p)`, if `rev = false`, where 's(j) = ±1'. 
The matrices `A(1)`, `A(2)`, `...`, `A(p)` are contained in the `n×n×p` array `A` 
such that the `i`-th matrix  `A(i)` is contained in `A[:,:,i]`. 

The resulting `n×n×p` array `S` contains the matrices `S(1)`, `...`, `S(p)` 
such that `S(ischur)` is in a quasi-upper triangular form, 
`S(i)` is upper triangular for `i` ``\\neq`` `ischur`. 
If `withZ = true` (default), the resulting `n×n×p` array `Z` contains the orthogonal transformation 
matrices `Z(1)`, `...`, `Z(p)`. The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 

The resulting matrices satisfy for `rev = true`

           Z(mod(j,p)+1)' * A(j) * Z(j) = S(j),  if S[j] = true, 
           Z(j)' * A(j) * Z(mod(j,p)+1) = S(j),  if S[j] = false, 

and for `rev = false`

           Z(j)' * A(j) * Z(mod(j,p)+1) = S(j),  if S[j] = true, 
           Z(mod(j,p)+1)' * A(j) * Z(j) = S(j),  if S[j] = false.


`S(i)` and `Z(i)` are contained in `S[:,:,i]` and `Z[:,:,i]`, respectively. 
The vector `ev` contains the eigenvalues of the appropriate matrix product. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 

The function `pgschur` is based on wrappers for the SLICOT subroutines `MB03VW`  
(see [`PeriodicSystems.SLICOTtools.mb03vw!`](@ref)) and `MB03BD`  (see [`PeriodicSystems.SLICOTtools.mb03bd!`](@ref)), 
based on algorithms proposed in [1] and [2].
   
REFERENCES

[1] Bojanczyk, A., Golub, G. H. and Van Dooren, P.
    The periodic Schur decomposition: algorithms and applications.
    In F.T. Luk (editor), Advanced Signal Processing Algorithms,
    Architectures, and Implementations III, Proc. SPIE Conference,
    vol. 1770, pp. 31-42, 1992.

[2] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.
"""
function pgschur(A::AbstractArray{Float64,3}, S::Union{AbstractVector{Bool},BitVector}; kwargs...)
   pgschur!(copy(A), S; kwargs...)
end
"""
    pgschur!(A::Array{Float64,3}, S::Union{Vector{Bool},BitVector}; kwargs...)

Same as `pgschur` but uses the input matrix `A` as workspace.
"""
function pgschur!(A::AbstractArray{Float64,3}, S::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, withZ::Bool = true, schurindex::Int = 1)
   n = size(A,1)
   n == size(A,2) || throw(DimensionMismatch("A must have equal first and second dimensions"))
   p = size(A,3)
   p == length(S) || throw(DimensionMismatch("length of S must be equal to the third dimension of A"))
   1 <= schurindex <= p ||  throw(ArgumentError("schurindex out of allowed range [1,$p]"))
   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n; 
   
   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'I'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   if rev
      reverse!(A, dims = 3)
      SIND = reverse(2*S.-1)
      hc = p - schurindex + 1
   else
      SIND = 2*S.-1
      hc = schurindex
   end

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 

   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, A, Z, LIWORK, LDWORK)

   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, hc, ilo, ihi, SIND, A, Z, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI) ./ BETA
   γ = 2. .^SCAL
   ev = α .* γ
   
   if rev
      return reverse!(A, dims = 3), withZ ? Z[:,:,mod.(p:-1:1,p).+1] : nothing, ev, schurindex, α, γ
   else
      return A, withZ ? Z : nothing, ev, schurindex, α, γ
   end
end
"""
     psordschur!(S::Vector{Matrix{Float64}}, Z::Vector{Matrix{Float64}}, select; rev, schurindex) 

Reorder the core eigenvalues of the product `Π = S[p]*...*S[2]*S[1]`, if `rev = true` (default) or `Π = S[1]*S[2]*...*S[p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The `p`-vectors `S` and `Z` contain the matrices `S[1]`, `...`, `S[p]` in an extended periodic Schur form, with the leading square block of 
`S[schurindex]` in real Schur form, and the corresponding orthogonal transformation matrices `Z[1]`, `...`, `Z[p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.  The dimension of select must be equal to the number of
core eigenvalues (i.e., the minimum dimension of matrices in the vector `S`).  
"""
function psordschur!(S::AbstractVector{Matrix{Float64}}, Z::AbstractVector{Matrix{Float64}}, select::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, schurindex::Int = 1) 

   k = length(S) 
   m = size.(S,1); n = size.(S,2) 
   rev || (reverse!(m); reverse!(n))
   ldq = rev ? n : [m[end];m[1:end-1]]
   nc = minimum(n)
   kschur = rev ? schurindex : k-schurindex+1
   ni = zeros(Int,k)
   s = ones(Int,k)
   t = zeros(0)
   rev ? [push!(t,S[i][:]...) for i in 1:k] : [push!(t,S[k-i+1][:]...) for i in 1:k]
   ldt = m; ixt = [1;(cumsum(m.*n).+1)[1:end-1]]
   q = Z[1][:]
   rev ? [push!(q,Z[i][:]...) for i in 2:k] : [push!(q,Z[k-i+1][:]...) for i in 1:k-1]
   rev || (m1 = [m[end];m[1:end-1]])
   ixq = rev ? [1;(cumsum(n.*n).+1)[1:end-1]] : [1;(cumsum(m1.*m1).+1)[1:end-1]]
   tol = 20. 
   ldwork = max(42*k + max(nc,10), 80*k - 48) 

   _, info = mb03kd!('U','N', k, nc, kschur, n, ni, s, Int.(select), t, ldt, ixt, q, ldq, ixq, tol, ldwork)

   info == 1 && error("reordering failed because some eigenvalues are too close to separate") 
   

   rev ? [S[i] = reshape(view(t,ixt[i]:ixt[i]+m[i]*n[i]-1),m[i],n[i]) for i in 1:k] : [S[k-i+1] = reshape(view(t,ixt[i]:ixt[i]+m[i]*n[i]-1),m[i],n[i]) for i in 1:k] 
   rev ? [Z[i] = reshape(view(q,ixq[i]:ixq[i]+n[i]*n[i]-1),n[i],n[i]) for i in 1:k] : 
         (Z[1] = reshape(view(q,ixq[1]:ixq[1]+n[1]*n[1]-1),n[1],n[1]); [Z[k-i+2] = reshape(view(q,ixq[i]:ixq[i]+n[i]*n[i]-1),n[i],n[i]) for i in 2:k])

   return nothing
end

"""
     psordschur1!(S::Vector{Matrix{Float64}}, Z::Vector{Matrix{Float64}}, select; rev, schurindex) 

Reorder the eigenvalues of the product `Π = S[p]*...*S[2]*S[1]`, if `rev = true` (default) or `Π = S[1]*S[2]*...*S[p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The `p`-vectors `S` and `Z` contain, respectively, the square matrices `S[1]`, `...`, `S[p]` in a periodic Schur form, with `S[schurindex` in real Schur form, 
and the corresponding orthogonal transformation matrices `Z[1]`, `...`, `Z[p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function psordschur1!(S::AbstractVector{Matrix{Float64}}, Z::AbstractVector{Matrix{Float64}}, select::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, schurindex::Int = 1) 
   k = length(S) 
   m = size.(S,1); n = size.(S,2) 
   nc = n[1]; 
   (all(m .== nc) && all(n .== nc)) || error("all elements of S must be square matrices of same dimension")
   k == length(Z) || error("S and Z must have the same length")
   (all(size.(Z,1) .== nc) && all(size.(Z,2) .== nc)) || error("all elements of Z must be square matrices of same dimension as S")
   kschur = rev ? schurindex : k-schurindex+1
   ni = zeros(Int,k)
   s = ones(Int,k)
   t = zeros(0)
   rev ? [push!(t,S[i][:]...) for i in 1:k] : [push!(t,S[k-i+1][:]...) for i in 1:k]
   nn = nc*nc
   ldt = n; ixt = collect(1:nn:k*nn)
   q = Z[1][:]
   rev ? [push!(q,Z[i][:]...) for i in 2:k] : [push!(q,Z[k-i+1][:]...) for i in 1:k-1]
   ldq = ldt; ixq = ixt;
   tol = 20. 
   ldwork = max(42*k + max(nc,10), 80*k - 48) 

   _, info = mb03kd!('U','N', k, nc, kschur, n, ni, s, Int.(select), t, ldt, ixt, q, ldq, ixq, tol, ldwork)

   info == 1 && error("reordering failed because some eigenvalues are too close to separate") 
   

   rev ? [S[i] = reshape(view(t,ixt[i]:ixt[i]+nn-1),nc,nc) for i in 1:k] : [S[k-i+1] = reshape(view(t,ixt[i]:ixt[i]+nn-1),nc,nc) for i in 1:k] 
   rev ? [Z[i] = reshape(view(q,ixq[i]:ixq[i]+nn-1),nc,nc) for i in 1:k] : 
         (Z[1] = reshape(view(q,ixq[1]:ixq[1]+nn-1),nc,nc); [Z[k-i+2] = reshape(view(q,ixq[i]:ixq[i]+nn-1),nc,nc) for i in 2:k])

   return nothing
end
"""
     psordschur!(S::Array{Float64,3}, Z::Array{Float64,3}, select::Union{Vector{Bool},BitVector}; rev, schurindex) 

Reorder the eigenvalues of the product `Π = S[:,:,p]*...*S[:,:,2]*S[:,:,1]`, if `rev = true` (default) or `Π = S[:,:,1]*S[:,:,2]*...*S[:,:,p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The 3-dimensional arrays `S` and `Z` contain the matrices `S[:,:,1]`, `...`, `S[:,:,p]` in a periodic Schur form, with `S[:,:,schurindex]` in real Schur form, 
and the corresponding orthogonal transformation matrices `Z[:,:,1]`, `...`, `Z[:,:,p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function psordschur!(S::Array{Float64,3}, Z::Array{Float64,3}, select::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, schurindex::Int = 1) 

   mc, nc, k = size(S) 
   mc == nc || error("S must have the same first and second dimensions")
   mz, nz, kz = size(Z) 
   k == kz || error("S and Z must have the same length")
   mz == nz == nc || error("Z must have the same first and second dimensions as S")
   kschur = rev ? schurindex : k-schurindex+1
   ni = zeros(Int,k)
   s = ones(Int,k)
   if rev
      t = view(S,:); q = view(Z,:)
   else
      t = zeros(0); [push!(t,S[:,:,k-i+1][:]...) for i in 1:k]
      # rev ? [push!(t,S[:,:,i][:]...) for i in 1:k] : [push!(t,S[:,:,k-i+1][:]...) for i in 1:k]
      q = Z[:,:,1][:]; [push!(q,Z[:,:,k-i+1][:]...) for i in 1:k-1]
      #rev ? [push!(q,Z[:,:,i][:]...) for i in 2:k] : [push!(q,Z[:,:,k-i+1][:]...) for i in 1:k-1]
   end
   nn = nc*nc
   ldt = nc*ones(Int,k); ixt = collect(1:nn:k*nn)
   ldq = ldt; ixq = ixt;
   tol = 20. 
   ldwork = max(42*k + max(nc,10), 80*k - 48) 

   _, info = mb03kd!('U','N', k, nc, kschur, ldt, ni, s, Int.(select), t, ldt, ixt, q, ldq, ixq, tol, ldwork)

   info == 1 && error("reordering failed because some eigenvalues are too close to separate") 
   
   if !rev
      [S[:,:,k-i+1] = reshape(view(t,ixt[i]:ixt[i]+nn-1),nc,nc) for i in 1:k] 
      Z[:,:,1] = reshape(view(q,ixq[1]:ixq[1]+nn-1),nc,nc); [
      Z[:,:,k-i+2] = reshape(view(q,ixq[i]:ixq[i]+nn-1),nc,nc) for i in 2:k]
   end
   # rev ? [S[:,:,i] = reshape(view(t,ixt[i]:ixt[i]+nn-1),nc,nc) for i in 1:k] : [S[:,:,k-i+1] = reshape(view(t,ixt[i]:ixt[i]+nn-1),nc,nc) for i in 1:k] 
   # rev ? [Z[:,:,i] = reshape(view(q,ixq[i]:ixq[i]+nn-1),nc,nc) for i in 1:k] : 
   #       (Z[:,:,1] = reshape(view(q,ixq[1]:ixq[1]+nn-1),nc,nc); [Z[:,:,k-i+2] = reshape(view(q,ixq[i]:ixq[i]+nn-1),nc,nc) for i in 2:k])

   return nothing
end
"""
     pgordschur!(S::Array{Float64,3}, s::Union{Vector{Bool},BitVector}, Z::Array{Float64,3}, select::Union{Vector{Bool},BitVector}; rev, schurindex) 

Reorder the eigenvalues of the product `Π = S[:,:,p]^s[p]*...*S[:,:,2]^s[2]*S[:,:,1]^s[1]`, if `rev = true` (default) or 
`Π = S[:,:,1]^s[1]*S[:,:,2]^s[2]*...*S[:,:,p]^s[p]` if `rev = false`, with 's[j] = ±1', where `Π` is in a real Schur form, 
such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The 3-dimensional arrays `S` and `Z` contain the matrices `S[:,:,1]`, `...`, `S[:,:,p]` in a generalized periodic Schur form, 
with `S[:,:,schurindex]` in a quasi-upper triangular (real Schur) form, 
and the corresponding orthogonal transformation matrices `Z[:,:,1]`, `...`, `Z[:,:,p]`, respectively.  
`S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function pgordschur!(S::AbstractArray{Float64,3}, s::Union{AbstractVector{Bool},BitVector}, Z::AbstractArray{Float64,3}, select::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, schurindex::Int = 1) 

   mc, nc, k = size(S) 
   mc == nc || error("S must have the same first and second dimensions")
   mz, nz, kz = size(Z) 
   k == kz || error("S and Z must have the same length")
   mz == nz == nc || error("Z must have the same first and second dimensions as S")
   kschur = rev ? schurindex : k-schurindex+1
   ni = zeros(Int,k)
   sarg = 2*s.-1
   #s = ones(Int,k)
   if !rev
      reverse!(S, dims=3)
      reverse!(view(Z,:,:,2:k), dims = 3)
      reverse!(sarg)
   end
   t = view(S,:); q = view(Z,:)
   nn = nc*nc
   ldt = nc*ones(Int,k); ixt = collect(1:nn:k*nn)
   ldq = ldt; ixq = ixt;
   tol = 20. 
   ldwork = max(42*k + max(nc,10), 80*k - 48) 

   _, info = mb03kd!('U','N', k, nc, kschur, ldt, ni, sarg, Int.(select), t, ldt, ixt, q, ldq, ixq, tol, ldwork)

   info == 1 && error("reordering failed because some eigenvalues are too close to separate") 
   
   if !rev
      reverse!(S,dims=3)
      reverse!(view(Z,:,:,2:k), dims = 3)
   end

   return nothing
end

"""
     pgordschur!(S::Vector{Matrix{Float64}}, s::Union{Vector{Bool},BitVector}, Z::Vector{Matrix{Float64}}, select::Union{Vector{Bool},BitVector}; rev, schurindex) 

Reorder the eigenvalues of the product `Π = S[p]^s[p]*...*S[2]^s[2]*S[1]^s[1]`, if `rev = true` (default) or 
`Π = S[1]^s[1]*S[2]^s[2]*...*S[p]^s[p]` if `rev = false`, with 's[j] = ±1', where `Π` is in a real Schur form, 
such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The `p`-vectors `S` and `Z` contain the matrices `S[1]`, `...`, `S[p]` in a generalized periodic Schur form, 
with `S[schurindex]` in a quasi-upper triangular (real Schur) form, 
and the corresponding orthogonal transformation matrices `Z[1]`, `...`, `Z[p]`, respectively.  
`S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function pgordschur!(S::AbstractVector{Matrix{Float64}}, s::Union{AbstractVector{Bool},BitVector}, Z::AbstractVector{Matrix{Float64}}, select::Union{AbstractVector{Bool},BitVector}; rev::Bool = true, schurindex::Int = 1) 
   k = length(S) 
   k == length(s) || error("S and s must have the same length")
   k == length(Z) || error("S and Z must have the same length")

   m = size.(S,1); n = size.(S,2) 
   rev || (reverse!(m); reverse!(n))
   ldq = rev ? n : [m[end];m[1:end-1]]
   nc = minimum(n)
   nc == length(select) || error("select must have length $nc")
   kschur = rev ? schurindex : k-schurindex+1
   ni = zeros(Int,k)
   sarg = 2*s.-1
   rev || reverse!(sarg)
   t = zeros(0)
   rev ? [push!(t,S[i][:]...) for i in 1:k] : [push!(t,S[k-i+1][:]...) for i in 1:k]
   ldt = m; ixt = [1;(cumsum(m.*n).+1)[1:end-1]]
   q = Z[1][:]
   rev ? [push!(q,Z[i][:]...) for i in 2:k] : [push!(q,Z[k-i+1][:]...) for i in 1:k-1]
   rev || (m1 = [m[end];m[1:end-1]])
   ixq = rev ? [1;(cumsum(n.*n).+1)[1:end-1]] : [1;(cumsum(m1.*m1).+1)[1:end-1]]
   tol = 20. 
   ldwork = max(42*k + max(nc,10), 80*k - 48) 

   _, info = mb03kd!('U','N', k, nc, kschur, n, ni, sarg, Int.(select), t, ldt, ixt, q, ldq, ixq, tol, ldwork)

   info == 1 && error("reordering failed because some eigenvalues are too close to separate") 
   

   rev ? [S[i] = reshape(view(t,ixt[i]:ixt[i]+m[i]*n[i]-1),m[i],n[i]) for i in 1:k] : [S[k-i+1] = reshape(view(t,ixt[i]:ixt[i]+m[i]*n[i]-1),m[i],n[i]) for i in 1:k] 
   rev ? [Z[i] = reshape(view(q,ixq[i]:ixq[i]+n[i]*n[i]-1),n[i],n[i]) for i in 1:k] : 
         (Z[1] = reshape(view(q,ixq[1]:ixq[1]+n[1]*n[1]-1),n[1],n[1]); [Z[k-i+2] = reshape(view(q,ixq[i]:ixq[i]+n[i]*n[i]-1),n[i],n[i]) for i in 2:k])

   return nothing
end
# similarity transformation checks
function check_psim(Ain::AbstractArray{T,3}, Q::AbstractArray{T,3}, Aout::AbstractArray{T,3}; rev::Bool = true, atol = 1.e-7) where T
   SSQ = 0.
   K = size(Ain,3)
   for i in 1:K
      # rev = true:  Q(i+1)' * Ain(i) * Q(i) = Aout(i),          
      # rev = false: Q(i)' * Ain(i) * Q(i+1) = Aout(i),
      Zta = rev ? view(Ain,:,:,i) * view(Q,:,:,i) - view(Q,:,:,mod(i,K)+1) * view(Aout,:,:,i) :
                  view(Ain,:,:,i) * view(Q,:,:,mod(i,K)+1) - view(Q,:,:,i) * view(Aout,:,:,i) 
      SSQ = hypot(SSQ, norm(Zta))
   end
   return SSQ < atol
end
function check_psim(Ain::Vector{Matrix{T}}, Q::Vector{Matrix{T1}}, Aout::Vector{Matrix{T1}}; rev::Bool = true, atol = 1.e-7) where {T,T1}
   SSQ = 0.
   K = length(Ain)
   for i in 1:K
      # rev = true:  Q(i+1)' * Ain(i) * Q(i) = Aout(i),          
      # rev = false: Q(i)' * Ain(i) * Q(i+1) = Aout(i),
      Zta = rev ? Ain[i] * Q[i] - Q[mod(i,K)+1] * Aout[i] :
                  Ain[i] * Q[mod(i,K)+1] - Q[i] * Aout[i] 
      SSQ = hypot(SSQ, norm(Zta))
   end
   return SSQ < atol
   # one line computation
   # return rev ? norm(norm.(Ain.*Q.-mshift(Q).*Aout)) < atol : norm(norm.(Ain.*mshift(Q).-Q.*Aout)) < atol
end
function mshift(X::Vector{<:Matrix}, k::Int = 1) 
   # Form a k-shifted array of matrices.
   K = length(X)
   return X[mod.(k:k+K-1,K).+1]
end

