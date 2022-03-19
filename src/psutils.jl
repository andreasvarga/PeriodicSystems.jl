"""
     Φ = tvstm(At::Function, tf, t0; solver, reltol, abstol, dt) 

Compute the state transition matrix for a linear ODE with time-varying coefficients. 
For the given time-dependent square matrix function `At`, initial time `t0` and 
final time `tf`, the state transition matrix `Φ(tf,t0)`
is computed by integrating numerically the homogeneous linear ODE 

      dΦ(t,t0)/dt = A(t)Φ(t,t0),  Φ(t0,t0) = I

on the time interval `[t0,tf]`. The ODE solver to be employed can be 
specified using the keyword argument `solver` (see below), together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = tf-t0`). 
Depending on the desired relative accuracy `reltol`, 
lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers form the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "linear"` - use a special solver for linear ODEs (`MagnusGL6()`) with fixed time step `dt`;

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
"""
function tvstm(At::Function, tf::Real, t0::Real = 0; solver = "", reltol = 1e-3, abstol = 1e-7, dt = (tf-t0)/10) 
   
   At0 = At(t0)
   n = LinearAlgebra.checksquare(At0)
   T = promote_type(eltype(At0), typeof(t0), typeof(tf))

   # using OrdinaryDiffEq
   u0 = Matrix{T}(I,n,n)
   tspan = (T(t0),T(tf))
   if solver != "linear" 
      f!(du,u,p,t) = mul!(du,At(t),u)
      prob = ODEProblem(f!, u0, tspan)
   end
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, save_evrystep = false)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, save_evrystep = false)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, save_evrystep = false)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, save_evrystep = false)
      end
   elseif solver == "linear" 
      function update_func!(A,u,p,t)
         A .= p(t)
      end
      DEop = DiffEqArrayOperator(ones(T,n,n),update_func=update_func!)     
      prob = ODEProblem(DEop, u0, tspan, At)
      sol = solve(prob,MagnusGL6(), dt = dt, save_evrystep = false)
   elseif solver == "symplectic" 
      # high accuracy symplectic
      sol = solve(prob, IRKGaussLegendre.IRKGL16(); reltol, abstol, save_evrystep = false)
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_evrystep = false)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5()); nonstifftol = 11/10, reltol, abstol, save_evrystep = false)
      end
   end

   return sol(tf)     
end
""" 
     monodromy(At, K = 1; solver, reltol, abstol, dt) -> Ψ 

Compute the monodromy matrix for a linear ODE with periodic time-varying coefficients. 

For the given square periodic function matrix `At` of period `T`, 
the monodromy matrix `Ψ = Φ(T,0)` is computed, where `Φ(t,τ)` is the state transition matrix satisfying the homogeneous linear ODE 

    dΦ(t,τ)/dt = A(t)Φ(t,τ),  Φ(τ,τ) = I. 

If `K > 1`, then `Ψ = Φ(T,0)` is determined as a product of `K` matrices 
`Ψ = Ψ_K*...*Ψ_1`, where for `Δ := T/K`, `Ψ_i = Φ(iΔ,(i-1)Δ)` is the 
state transition matrix on the time interval `[(i-1)Δ,iΔ]`. 
The resulting `Ψ` is a periodic array object. 

The state transition matrices `Φ(iΔ,(i-1)Δ)`
are computed by integrating numerically the above homogeneous linear ODE.  
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = tf-t0`) (see [`tvstm`](@ref)). 
For large values of `K`, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  
"""
function monodromy(at::PeriodicFunctionMatrix{:c,T}, K::Int = 1; solver = "non-stiff", reltol = 1e-3, abstol = 1e-7, dt = at.period/max(K,100)) where T
   n = LinearAlgebra.checksquare(at.f(0))
   Ts = at.period/K
   M = Array{T,3}(undef, n, n, K) 
   K >= 100 ? dt = Ts : dt = Ts*K/100

   Threads.@threads for i = 1:K
       @inbounds M[:,:,i] = tvstm(at.f, i*Ts, (i-1)*Ts; solver = solver, reltol = reltol, abstol = abstol, dt = dt) 
   end
   return PeriodicArray(M,at.period)
end
"""
     pseig(A, K = 1; lifting = false, solver, reltol, abstol, dt) -> ev

Compute the characteristic multipliers of a periodic matrix. 
For the given square continuous-time periodic matrix `A(t)` of period `T`, 
the characteristic multipliers `ev` are the eigenvalues of 
the monodromy matrix `Ψ = Φ(T,0)`, where `Φ(t,τ)` is the state transition matrix satisfying the homogeneous linear ODE 

    dΦ(t,τ)/dt = A(t)Φ(t,τ),  Φ(τ,τ) = I. 

If `lifting = false`, `Ψ` is computed as a product of `K` state transition matrices 
`Ψ = Ψ_K*...*Ψ_1` (see [`monodromy`](@ref) with the associated keyword arguments). 
The eigenvalues are computed using the periodic Schur decomposition method of [1].

If `lifting = false`, `Ψ` is determined as `Ψ = inv(N)*M`, where `M-λN` is a regular
pencil with `N` invertible and  
the eigenvalues of `M-λN` are the same as those of the matrix product
`Ψ := Ψ_K*...*Ψ_1`. 
An efficient version of the structure exploiting fast reduction method of [2] is employed, 
which embeds the determination of transition matrices into the reduction algorithms. 
This option may occasionally lead to inaccurate results for large values of `K`. 
`A` may be a _periodic function matrix object_, or a _periodic symbolic matrix object_, or a 
_harmonic representation object_ or a _periodic time series matrix object_.

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.

"""
function pseig(at::PeriodicFunctionMatrix{:c,T}, K::Int = 1; lifting::Bool = false, solver = "non-stiff", reltol = 1e-3, abstol = 1e-7, dt = at.period/100) where T
   n = LinearAlgebra.checksquare(at.f(0))
   t = 0  
   Ts = at.period/K
   if lifting 
      if K == 1
         ev = eigvals(tvstm(at.f, at.period, 0; solver, reltol, abstol, dt)) 
      else   
         si = tvstm(at.f, Ts, 0; solver, reltol, abstol); ti = -I
         t = Ts
         for i = 1:K-1
             tf = t+Ts
             F = qr([ ti; tvstm(at.f, tf, t; solver, reltol, abstol, dt) ])     
             si = F.Q'*[si; zeros(T,n,n)];  si = si[n+1:end,:]
             ti = F.Q'*[ zeros(T,n,n); -I]; ti = ti[n+1:end,:]
             t = tf
         end
         ev = -eigvals(si,ti)
      end
      sorteigvals!(ev)
   else
      M = monodromy(at, K; solver, reltol, abstol, dt) 
      ev = pschur(M.M; withZ = false)[3]
      return isreal(ev) ? real(ev) : ev
   end
end
pseig(at::PeriodicSymbolicMatrix{:c,T}, K::Int = 1; kwargs...) where T = 
    pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)
pseig(at::HarmonicArray{:c,T}, K::Int = 1; kwargs...) where T = 
    pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)
pseig(at::PeriodicTimeSeriesMatrix{:c,T}, K::Int = 1; kwargs...) where T = 
    pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)

"""
     pcseig(A, K = 1; lifting = false, solver, reltol, abstol, dt) -> ce

Compute the characteristic exponents of a periodic matrix.
For a given square continuous-time periodic function matrix `A(t)` of period `T`, 
the characteristic exponents `ce` are `log(ev)/T`, where  `ev` are the characteristic
multipliers (i.e., the eigenvalues of the monodromy matrix of `A(t)`).  
For available options see [`pseig(::PeriodicFunctionMatrix)`](@ref). 
For a given square discrete-time periodic matrix `A(t)` of discrete period `N`,  
the characteristic exponents `ce` are `ev.^-N`. 
"""
function psceig(at::AbstractPeriodicArray{:c}, K::Int = 1; kwargs...) where T
   ce = log.(complex(pseig(convert(PeriodicFunctionMatrix,at), K; kwargs...)))/at.period
   return isreal(ce) ? real(ce) : ce
end
function psceig(at::AbstractPeriodicArray{:d}) where T
   ce = (complex(pseig(convert(PeriodicMatrix,at)))).^-(at.dperiod) 
   return isreal(ce) ? real(ce) : ce
end
"""
     ev = pseig(A; rev = true, fast = false) 

Compute the eigenvalues of a product of `p` square matrices 
`A(p)...*A(2)*A(1)`, if `rev = true` (default) (also called characteristic multipliers) or 
of `A(1)*A(2)...A(p)` if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `n×n×p` array `A` 
such that the `i`-th matrix `A(i)` is contained in `A[:,:,i]`.
Alternatively, `A` can be a [`PeriodicArray`](@ref) object. 
If `fast = false` (default) then the eigenvalues are computed using an approach
based on the periodic Schur decomposition [1], while if `fast = true` 
the structure exploiting reduction [2] of an appropriate lifted pencil is employed.
This later option may occasionally lead to inaccurate results for large number of matrices. 

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.

"""
function pseig(A::Array{Float64,3}; rev::Bool = true, fast::Bool = false) where T
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   if fast 
      if rev 
         ev = eigvals(psreduc_reg(A)...)
      else
         imap = p:-1:1                     
         ev = eigvals(psreduc_reg(view(A,imap))...)
      end
      isreal(ev) && (ev = real(ev))
      sorteigvals!(ev)
      return sort!(ev,by=abs,rev=true)
   else
      ev = pschur(A; rev, withZ = false)[3]
      isreal(ev) && (ev = real(ev))
      return ev
   end
end
pseig(A::PeriodicArray{T}; fast::Bool = false) where T = pseig(A.M; fast)
"""
     ev = pseig(A; rev = true, fast = false, istart = 1) 

Compute the eigenvalues of a square cyclic product of `p` matrices 
`A(k-1)...*A(2)*A(1)*A(p)...*A(k)`, if `rev = true` (default) or 
`A(k)*A(k+1)*...A(p)*A(1)...A(k-1)` if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `p`-vector of matrices `A` 
such that the `i`-th matrix  `A(i)`, of dimensions `m(i)×n(i)`, is contained in `A[i]`.
Alternatively, `A` can be [`PeriodicMatrix`](@ref) object. 
`istart = k` specifies the starting index (default: `k = 1`). 
If `fast = false` (default) then the eigenvalues are computed using an approach
based on the periodic Schur decomposition [1], while if `fast = true` 
the structure exploiting reduction [2] of an appropriate lifted pencil is employed. 
This later option may occasionally lead to inaccurate results for large number of matrices. 

_Note:_ The first `nmin` components of `ev` contains the `core eigenvalues` of the appropriate matrix product,
where `nmin` is the minimum row dimensions of matrices `A[i]`, for `i = 1, ..., p`, 
while the last `ncur-nmin` components of `ev` are zero, 
where `ncur` is the column dimension of `A[k]` if `rev = true` or 
the row dimension of `A[k]` if `rev = false`. 

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.

"""
function pseig(A::Vector{Matrix{T}}; rev::Bool = true, fast::Bool = false, istart::Int = 1) where T
   p = length(A)
   istart = mod(istart-1,p)+1
   nev = rev ? size(A[istart],2) : size(A[istart],1)
   # check dimensions
   m, n = size.(A,1), size.(A,2)
   if rev
      all(m .== view(n,mod.(1:p,p).+1)) || error("incompatible dimensions")
   else
      all(n .== view(m,mod.(1:p,p).+1)) || error("incompatible dimensions")
   end
   if fast 
      ncore = minimum(size.(A,1))
      if istart == 1 && rev 
         ev = eigvals(psreduc_reg(A)...)
      else
         imap = rev ? (mod.(istart:istart+p-1,p).+1) :
                      (mod.(p-istart+1:-1:-istart+2,p).+1)                     
         ev = eigvals(psreduc_reg(view(A,imap))...)
      end
      isreal(ev) && (ev = real(ev))
      T <: Complex || sorteigvals!(ev)
      ind = sortperm(ev; by = abs, rev = true) # select the core eigenvalues
      return [ev[ind[1:ncore]]; zeros(eltype(ev),nev-ncore)]  # pad with the necessary zeros
   else
      if istart == 1 
         ev = pschur(A; rev, withZ = false)[3]
      else
         # avoid repeated reindexing
         imap = mod.(istart-1:istart+p-2,p).+1
         rev && reverse!(imap)        
         ev = pschur(view(A,imap); rev = false, withZ = false)[3]
      end
      isreal(ev) && (ev = real(ev))
      return ev[1:nev]
   end
end
pseig(A::PeriodicMatrix{T}; fast::Bool = false, istart::Int = 1) where T = pseig(A.M; fast, istart)

function sorteigvals!(ev)
   # an approximately complex conjugated set is assumed 
   isreal(ev) && (return ev)
   tc = ev[imag.(ev) .> 0]
   ev[:] = [ev[imag.(ev) .== 0]; sort([tc; conj.(tc)],by = real)]
   return ev
end

"""
     pschur(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

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
The first `nmin` components of `ev := α .* γ` contain the core eigenvalues of the appropriate matrix product,
where `nmin = m(ischur)`, while the last `nmax-nmin` components of `ev` are zero, 
where `nmax` is the largest row or column dimension of `A(i)`, for `i = 1, ..., p`. 
The eigenvalues can be alternatively expressed as `α .* γ`, where `γ` contains suitable 
scaling parameters to avoid overflows or underflows in the expressions of the eigenvalues. 
The performed orthogonal transformations are not accumulated if `withZ = false`, 
in which case `Z = nothing`. 
"""
function pschur(A::AbstractVector{Matrix{T}}; rev::Bool = true, withZ::Bool = true) where T

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
   
   St = zeros(T, n, n, p)
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
   α = complex.(ALPHAR, ALPHAI)
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
     pschur(A; sind = 1, rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

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
"""
function pschur(A::Array{Float64,3}; rev::Bool = true, sind::Int = 1, withZ::Bool = true)
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
   α = complex.(ALPHAR, ALPHAI)
   γ = 2. .^SCAL
   ev = α .* γ

   if shift
     if rev
        return S[:,:,imap], withZ ? Z[:,:,mod.(p+sind:-1:sind+1,p).+1] : nothing, ev, sind, α, γ
     else
        # return back shifted matrices
        imap1 = mod.(imap.+(sind+1),p).+1
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
"""
     phess(A; hind = 1, rev = true, withZ = true) -> (H, Z, ihess)

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
"""
function phess(A::AbstractArray{Float64,3}; rev::Bool = true, hind::Int = 1, withZ::Bool = true)
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
          imap1 = mod.(imap.+(hind+1),K).+1
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


# conversions
"""
     ts2pfm(At::PeriodicTimeSeriesMatrix; method = "linear") -> A::PeriodicFunctionMatrix

Compute the periodic function matrix corresponding to an interpolated periodic time series matrix. 
For the given periodic time series matrix `At`, a periodic function matrix `A(t)` is defined as the 
mapping `A(t) = t -> etpf(t)`, where `etpf(t)` is a periodic interpolation/extrapolation object,  
as provided in the `Interpolation.jl` package. The keyword parameter method specifies the
interpolation/extrapolation method to be used as follows:

`method = "constant"` - use periodic B-splines of degree 0 

`method = "linear"` - use periodic B-splines of degree 1 (periodic linear interpolation; default)

`method = "quadratic"` - use periodic B-splines of degree 2 (periodic quadratic interpolation) 

`method = "cubic"` - use periodic B-splines of degree 3 (periodic cubic interpolation) 
"""
function ts2pfm(A::PeriodicTimeSeriesMatrix; method = "linear")
   N = length(A.values)
   N == 0 && error("empty time array not supported")
   N == 1 && (return t -> A.values[1])
   ts = (0:N-1)*(A.period/N)
   n1, n2 = size(A.values[1])
   intparray = Array{Any,2}(undef,n1, n2)
   if method == "linear"
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Linear(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "cubic"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Cubic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "quadratic"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Quadratic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "constant"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Constant(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   else
      error("no such option method = $method")
   end
   return PeriodicFunctionMatrix(t -> [intparray[i,j](t) for i in 1:n1, j in 1:n2 ], A.period )
end

"""
     ts2hr(A::PeriodicTimeSeriesMatrix; atol = 0, rtol = √ϵ, n) -> Ahr::HarmonicArray

Compute the harmonic (Fourier) approximation of a periodic matrix specified by a time series. 
The periodic matrix `A(t)` is specified as a continuous-time periodic time series matrix `A`, 
with `m` matrices contained in the vector of matrices `A.values`, where `A.values[k]` 
is the value of `A(t)` at time moment `(k-1)T/m`, with `T = A.period` being the period. 
The resulting harmonic approximation `Ahr(t)` of `A(t)` has the form

                     p
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T)+As_i*sin(i*2*π*t/T) ) 
                    i=1 

where `A_0` is the constant term (the mean value), `Ac_i` and `As_i` are the  
coefficient matrices of the `i`-th cosinus and sinus terms, respectively. 
The order of the approximation `p` is determined using the maximum order specified by `n` 
(default: `n = (m-1)/2`) and the absolute and relative tolerances `atol` and `rtol`, as follows:
`p` is the minimum between `n`, `(m-1)/2` and the maximum index `k` 
such that `Ac_k` and/or `As_k` are nonzero.
The tolerance used to assess nonzero elements is `tol = max(atol, rtol*maxnorm)`, where 
`maxnorm` is the maximum norm of the matrices contained in `A.values`. The default values of tolerances
are `atol = 0` and `rtol = √ϵ`, where `ϵ` is the working machine precision.

The resulting harmonic approximation `Ahr(t)` is returned in the harmonic array object `Ahr` 
(see [`HarmonicArray`](@ref)). 
"""
function ts2hr(A::PeriodicTimeSeriesMatrix{:c,T}; atol::Real = 0, rtol::Real = 0, n::Union{Int,Missing} = missing) where  {T, Ts}
   
   M = length(A.values)
   n1, n2 = size(A.values[1])
   
   if ismissing(n)
       n = Int(floor((M-1)/2))
       ncur = 0
   else
       n = min( n, Int(floor((M-1)/2)) ) 
       ncur = n
   end
   n = max(n,0)
   
   AHR = zeros(ComplexF64, n1, n2, n+1)
   tol = iszero(atol) ? (iszero(rtol) ? 10*M*eps(maximum(norm.(A.values))) : rtol*maximum(norm.(A.values)) ) : atol
   i1 = 1:n+1   
   for i = 1:n1
       for j = 1:n2
           temp = getindex.(A.values, i, j)
           i == 1 && j == 1 && (global rfftop = plan_rfft(temp; flags=FFTW.ESTIMATE, timelimit=Inf))
           tt = conj(2/M*(rfftop*temp)) 
           tt[1] = real(tt[1]/2)
           tt1 = view(tt,i1)
           indr = i1[abs.(real(tt1)) .> tol]
           nr = length(indr); 
           nr > 0 && (nr = indr[end])
           indi = i1[abs.(imag(tt1)) .> tol]
           ni = length(indi); 
           ni > 0 && (ni = indi[end])
           ncur = max(ncur, nr, ni)        
           AHR[i,j,indr] = real(tt[indr])
           AHR[i,j,indi] += im*imag(tt[indi])
       end
   end
   return HarmonicArray(AHR[:,:,1:ncur],A.period)
end

"""
   hr2psm(Ahr::HarmonicArray, nrange) -> A::Matrix

Convert a range of harmonic components `nrange` of the harmonic array `Ahr` to a symbolic matrix `A`. 
The default range is `nrange = 0:n`, where `n` is the order of the maximum harmonics.  
"""
function hr2psm(ahr::HarmonicArray, nrange::UnitRange = 0:size(ahr.values,3)-1)   
   Symbolics.@variables t   
   Period = ahr.period
   i1 = first(nrange)
   i2 = last(nrange)
   (i1 < 0 || i2 >= size(ahr.values,3)) && error("nrange out of allowed range")
   
   ts = t*2*pi/Period
   
   i1 == 0 ? a = Num.(real.(ahr.values[:,:,1])) : a = zeros(Num,size(ahr.values,1),size(ahr.values,2))
   for i in i1+1:i2
       ta = view(ahr.values,:,:,i+1)
       a .+= real.(ta).*(cos(i*ts)) .+ imag.(ta) .* (sin(i*ts))
   end
   return a
end   


# lifting-based structure exploiting (fast) reductions

# function ps2ls(psys::PeriodicDiscreteDescriptorStateSpace{T}; kstart::Int = 1, compacted::Bool = false, 
#              fast::Bool = true, atol::Real = zero(real(T)), atol1::Real = atol, atol2::Real = atol, 
#              rtol::Real = eps(real(float(one(T))))*iszero(max(atol1,atol2))) where T
#     indp(j,n) = mod(j-1,n)+1
#     desc = (psys.E != I)
#     K = psys.period
#     Ts = psys.Ts 
#     LTs = Ts > 0 ? Ts*K : -1
#     #[ap,bp,cp,dp,ep] = pdsdata(psys);
#     # generate the stacked lifted system of (Grasselli and Longhi, 1991)
#     kind = indp.(kstart:kstart+K-1,K)
#     ni = psys.nx[kind]
#     mui = psys.ndx[kind]
#     m = psys.nu[kind]
#     n = sum(ni)
#     mukm1 = sum(view(mui,1:K-1))
#     nukm1 = sum(view(ni,1:K-1))
#     KK = kind[K]
#     kindm1 = view(kind,1:K-1)
#     A = [zeros(T, mui[K], nukm1) psys.A[KK]; blockdiag(view(psys.A,kindm1)...) zeros(T, mukm1, ni[K])] +
#         [zeros(T, mui[K], n); zeros(T, mukm1, ni[1]) desc ? -blockdiag(view(psys.E,kindm1)...) : -I]
#     E = [desc ? psys.E[KK] : I zeros(T, mui[K], n-ni[1]); zeros(T, mukm1, n)]
#     B = [ zeros(T, mui[K], sum(view(m,1:K-1))) psys.B[KK]; blockdiag(view(psys.B,kindm1)...) zeros(T, mukm1, m[K])]
#     C = blockdiag(view(psys.C,kind)...) 
#     D = blockdiag(view(psys.D,kind)...) 
#     if compacted 
#        A, E, B, C, D = lsminreal(A, E, B, C, D; fast, atol1, atol2, rtol, contr = false, obs = false, noseig = true) 
#     end
#     return dss(A, E, B, C, D; Ts = LTs)
# end

function psreduc_fast(S::Vector{Matrix{T1}}, T::Vector{Matrix{T1}}; atol::Real = 0) where T1
    # PSREDUC_FAST  Finds a matrix pair having the same finite and infinite 
    #               eigenvalues as a given periodic pair.
    #
    # [A,E] = PSREDUC_FAST(S,T,tol) computes a matrix pair (A,E) having 
    # the same finite and infinite eigenvalues as the periodic pair (S,T).
    #
    
    #   A. Varga 30-3-2004. 
    #   Revised: .
    #
    #   Reference:
    #   A. Varga & P. Van Dooren
    #      Computing the zeros of periodic descriptor systems.
    #      Systems and Control Letters, vol. 50, 2003.
    
    K = length(S) 
    if K == 1
       return S[1], T[1] 
    else
       m = size.(S,1)
       n = size.(S,2)
       if sum(m) > sum(n) 
           # build the dual pair
        #    [S,T]=celltranspose(S,T); 
        #    [S{1:K}] = deal(S{K:-1:1});
        #    [T{1:K-1}] = deal(T{K-1:-1:1});
        #    m = size.(S,1)
        #    n = size.(S,2)
       end 
    
       si = S[1];  ti = -T[1];
       tolr = atol
       for i = 1:K-2
           F = qr([ ti; S[i+1] ], ColumnNorm()) 
           nr = minimum(size(F.R))
           # compute rank of r 
           ss = abs.(diag(F.R[1:nr,1:nr]))
           atol == 0 && ( tolr = (m[i]+m[i+1]) * maximum(ss) * eps())
           rankr = count(ss .> tolr)
    
           si = F.Q'*[si; zeros(T1,m[i+1],n[1])]; si=si[rankr+1:end,:]
           ti = F.Q'*[ zeros(T1,size(ti,1),n[i+2]); -T[i+1]]; ti = ti[rankr+1:end,:]
       end
       a = [ zeros(T1,m[K],n[1]) S[K]; si ti] 
       e = [ T[K] zeros(T1,m[K],n[K]); zeros(size(si)...) zeros(size(ti)...)] 
       return a, e
    end
end
"""
    psreduc_reg(A) -> (M, N)

Determine for a `n×n×p` array `A`, the matrix pair `(M, N)` 
with `N` invertible and `M-λN` regular, such that 
the eigenvalues of `M-λN` are the same as those of the matrix product
`A(p)*A(p-1)*...*A(1)`, where `A(i)` is contained in `A[:,:,i]`. 
The structure exploiting fast reduction method of [1] is employed.

[1] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.
"""
function psreduc_reg(A::AbstractArray{T,3}) where T
     
    K = size(A,3) 
    n = size(A,1)
    n == size(A,2) || error("A must have equal first and second dimensions")
    if K == 1
       return A[:,:,1], Matrix{T}(I, n, n)
    else   
       si = A[:,:,1];  ti = -I
       for i = 1:K-1
           F = qr([ ti; A[:,:,i+1] ])     
           si = F.Q'*[si; zeros(T,n,n)];  si = si[n+1:end,:]
           ti = F.Q'*[ zeros(T,n,n); -I]; ti = ti[n+1:end,:]
       end
       return si, -ti
    end
end
"""
    psreduc_reg(A) -> (M, N)

Determine for a `p`-dimensional vector of rectangular matrices `A`, 
the matrix pair `(M, N)` with `N` invertible and `M-λN` regular, such that 
the eigenvalues of `M-λN` are the same as those of the square 
matrix product `A(p)*A(p-1)*...*A(1)`, where `A(i)` is contained in `A[i]`. 
The structure exploiting fast reduction method of [1] is employed.

[1] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.
"""
function psreduc_reg(A::AbstractVector{Matrix{T}}) where T
     
   K = length(A) 
   n = size.(A,2) 
   n == size.(A,1)[mod.(2:K+1,K).+1] || 
      error("the number of column of A[i] must be equal to the number of rows of A[i+1]")

   if K == 1
      return A[1], Matrix{T}(I, n[1], n[1])
   else
      si = A[1];  ti = -I
      for i = 1:K-1
          F = qr([ ti; A[i+1] ])  
          mi1 = n[mod(i+1,K)+1]
          si = F.Q'*[si; zeros(T,mi1,n[1])]; si = si[n[i+1]+1:end,:] 
          ti = F.Q'*[ zeros(T,n[i+1],mi1); -I]; ti = ti[n[i+1]+1:end,:] 
      end
      return si, -ti
   end
end

# time response evaluations

"""
     tvmeval(At::PeriodicTimeSeriesMatrix, t; method = "linear") -> A::Vector{Matrix}

Time response evaluation of a periodic time series matrix.
For the periodic time series matrix `At` and the vector of time values `t`, 
an interpolation/extrapolation based approximation  
`A[i]` is evaluated for each time value `t[i]`. The keyword parameter method specifies the
interpolation/extrapolation method to be used as follows:

`method = "constant"` - use periodic B-splines of degree 0 

`method = "linear"` - use periodic B-splines of degree 1 (periodic linear interpolation) 

`method = "quadratic"` - use periodic B-splines of degree 2 (periodic quadratic interpolation) 

`method = "cubic"` - use periodic B-splines of degree 3 (periodic cubic interpolation) 

"""
function tvmeval(A::PeriodicTimeSeriesMatrix{:c,T}, t::Union{Real,Vector{<:Real}}; method = "linear") where T
   N = length(A.values)
   N == 0 && error("empty time array not supported")
   isa(t,Real) ? te = [t] : te = t
   N == 1 && (return [A.values[1] for i in te])
   nt = length(te)
   nt == 0 && (return zeros(T,size(A.values,1),size(A.values,2),0))
   dt = A.period/N
   ts = (0:N-1)*dt
   n1, n2 = size(A.values[1])
   intparray = Array{Any,2}(undef,n1, n2)
   if method == "linear"
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Linear(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "cubic"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Cubic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "quadratic"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Quadratic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "constant"      
      [intparray[i,j] = scale(extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Constant(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   else
      error("no such option method = $method")
   end
   return [[intparray[i,j].(te[k]) for i in 1:n1, j in 1:n2 ] for k = 1:nt ]
end
"""
     hreval(Ahr::HarmonicArray, t; ntrunc, exact = true) -> A::Matrix

Evaluate the harmonic array `Ahr` representing a continuous-time 
time periodic matrix `A(t)` for a numerical or symbolic time value `t`. 
For real `t`, if `exact = true (defualt)` an exact evaluation is computed, while for `exact = false`, 
a linear interpolation based approximation is computed (potentially more accurate in intersample points).
The keyword argument `ntrunc` specifies the number of harmonics to be used for the evaluation 
(default: maximum possible number). 
"""
function hreval(ahr::HarmonicArray, t::Union{Num,Real}; exact::Bool = true, ntrunc::Int = max(size(ahr.values,3)-1,0))   
      (ntrunc < 0 || ntrunc >= size(ahr.values,3)) && error("ntrunc out of allowed range")
   isa(t,Num) && (return hr2psm(ahr))
   n = ntrunc

     
   ts = mod(t,ahr.period)*2*pi/ahr.period
   
   # determine interpolation coefficients
   ht = ones(Float64,n);
   if !exact
      # use linear interpolation
      for i = 2:n
           x = pi*(i-1)/n
           ht[i] = (sin(x)/x)^2
      end
   end
   a = real.(ahr.values[:,:,1])
   for i = 1:n
       ta = view(ahr.values,:,:,i+1)
       a .+= real.(ta).*(cos(i*ts)*ht[i]) .+ imag.(ta) .* ((sin(i*ts)*ht[i]))
   end
   return a
end   
"""
     tvmeval(Ahr::HarmonicArray, t; ntrunc, exact = true) -> A::Vector{Matrix}

Time response evaluation of a harmonic array.
For the harmonic array `Ahr` representing a time periodic matrix A(t) and the vector of time values `t`, 
a exact evaluation or linear interpolation based approximation   
`A[i] = A(t[i])` is evaluated for each time value `t[i]`. 
If `exact = true (defualt)` an exact evaluation is computed, while for `exact = false`, 
a linear interpolation based approximation is computed (potentially more accurate in intersample points).
The keyword argument `ntrunc` specifies the number of harmonics to be used for evaluation 
(default: maximum possible number). 
"""
function tvmeval(ahr::HarmonicArray, t::Union{Real,Vector{<:Real}}; ntrunc::Int = size(ahr.values,3), 
                exact::Bool = true) 
       
   n = min(size(ahr.values,3)-1,ntrunc);
   
   isa(t,Real) ? te = [t] : te = t
   nt = length(te)
   period = ahr.period
   
   tscal = 2*pi/period
   # determine interpolation coefficients
   ht = ones(Float64,n);
   if !exact
      # use linear interpolation
      for i = 2:n
           x = pi*(i-1)/n
           ht[i] = (sin(x)/x)^2
      end
   end
   A = similar(Vector{Matrix}, nt)
   for j = 1:nt
       A[j] = real(ahr.values[:,:,1])
       tsj = mod(te[j],period)*tscal
       for i = 1:n
           ta = view(ahr.values,:,:,i+1)
           A[j] .+= real.(ta).*(cos(i*tsj)*ht[i]) .+ imag.(ta) .* ((sin(i*tsj)*ht[i]))
       end
   end
   return A
end   
"""
     tvmeval(Asym::PeriodicSymbolicMatrix, t) -> A::Vector{Matrix}

Time response evaluation of a periodic symbolic matrix.
For the periodic symbolic matrix `Asym` representing a time periodic matrix A(t) and the vector of time values `t`, 
`A[i] = A(t[i])` is evaluated for each time value `t[i]`. 
"""
function tvmeval(A::PeriodicSymbolicMatrix, t::Union{Real,Vector{<:Real}} )
   te = isa(t,Real) ? [mod(t,A.period)] : mod.(t,A.period)
   return (convert(PeriodicFunctionMatrix,A).f).(te)
end
"""
     tvmeval(Af::PeriodicFunctionMatrix, t) -> A::Vector{Matrix}

Time response evaluation of a periodic function matrix.
For the periodic function matrix `Af` representing a time periodic matrix A(t) and the vector of time values `t`, 
`A[i] = A(t[i])` is evaluated for each time value `t[i]`. 
"""
function tvmeval(A::PeriodicFunctionMatrix, t::Union{Real,Vector{<:Real}} )
   te = isa(t,Real) ? [mod(t,A.period)] : mod.(t,A.period)
   return (A.f).(te)
end
