"""
     tvstm(A, tf, t0; solver, reltol, abstol, dt) -> Φ 

Compute the state transition matrix for a linear ODE with periodic time-varying coefficients. 
For the given periodic square matrix `A(t)`, initial time `t0` and 
final time `tf`, the state transition matrix `Φ(tf,t0)`
is computed by integrating numerically the homogeneous linear ODE 

      dΦ(t,t0)/dt = A(t)Φ(t,t0),  Φ(t0,t0) = I

on the time interval `[t0,tf]`. `A(t)` can be specified as a PeriodicFunctionMatrix, HarmonicArray or FourierFunctionMatrix. 

The ODE solver to be employed can be 
specified using the keyword argument `solver` (see below), together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = tf-t0`). 
Depending on the desired relative accuracy `reltol`, 
lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "linear"` - use a special solver for linear ODEs (`MagnusGL6()`) with fixed time step `dt`;

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
"""
function tvstm(A::PM, tf::Real, t0::Real = 0; solver = "", reltol = 1e-3, abstol = 1e-7, dt = (tf-t0)/10) where 
         {T, PM <: Union{PeriodicFunctionMatrix{:c,T},HarmonicArray{:c,T},FourierFunctionMatrix{:c,T}}} 
   n = size(A,1)
   n == size(A,2) || error("the function matrix must be square")

   isconstant(A) && ( return exp(tpmeval(A,t0)*(tf-t0)) )
   
   T1 = promote_type(typeof(t0), typeof(tf))

   # using OrdinaryDiffEq
   u0 = Matrix{T}(I,n,n)
   tspan = (T1(t0),T1(tf))
   if solver != "linear" 
      LPVODE!(du,u,p,t) = mul!(du,tpmeval(A,t),u)
      prob = ODEProblem(LPVODE!, u0, tspan)
   end
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, save_everystep = false)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, save_everystep = false)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, save_everystep = false)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, save_everystep = false)
      end
   elseif solver == "linear" 
      function update_func!(A,u,p,t)
         A .= p(t)
      end
      DEop = DiffEqArrayOperator(ones(T,n,n),update_func=update_func!)     
      #prob = ODEProblem(DEop, u0, tspan, A.f)
      prob = ODEProblem(DEop, u0, tspan, t-> tpmeval(A,t))
      sol = solve(prob,MagnusGL6(), dt = dt, save_everystep = false)
   elseif solver == "symplectic" 
      # high accuracy symplectic
      sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = true, reltol, abstol, save_everystep = false)
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
      end
   end

   return sol(tf)     
end
""" 
     monodromy(A[, K = 1]; solver, reltol, abstol, dt) -> Ψ::PeriodicArray 

Compute the monodromy matrix for a linear ODE with periodic time-varying coefficients. 

For the given square periodic matrix `A(t)` of period `T` and subperiod `T′ = T/k`, where 
`k` is the number of subperiods,  
the monodromy matrix `Ψ = Φ(T′,0)` is computed, where `Φ(t,τ)` is the state transition matrix satisfying the homogeneous linear ODE 

    dΦ(t,τ)/dt = A(t)Φ(t,τ),  Φ(τ,τ) = I. 

`A(t)` can be specified as a PeriodicFunctionMatrix, HarmonicArray or FourierFunctionMatrix. 

If `K > 1`, then `Ψ = Φ(T′,0)` is determined as a product of `K` matrices 
`Ψ = Ψ_K*...*Ψ_1`, where for `Δ := T′/K`, `Ψ_i = Φ(iΔ,(i-1)Δ)` is the 
state transition matrix on the time interval `[(i-1)Δ,iΔ]`. 

The state transition matrices `Φ(iΔ,(i-1)Δ)`
are computed by integrating numerically the above homogeneous linear ODE.  
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = min(Δ, Δ*K′/100)`) (see [`tvstm`](@ref)). 
For large values of `K`, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  
"""
function monodromy(A::PM, K::Int = 1; solver = "non-stiff", reltol = 1e-3, abstol = 1e-7, dt = A.period/max(K,100)) where
         {T, PM <: Union{PeriodicFunctionMatrix{:c,T},HarmonicArray{:c,T},FourierFunctionMatrix{:c,T}}} 
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix must be square")
   nperiod = A.nperiod
   Ts = A.period/K/nperiod

   M = Array{float(T),3}(undef, n, n, K) 

   # compute the matrix exponential for K = 1 and constant matrix
   K == 1 && isconstant(A) && ( M[:,:,1] = exp(tpmeval(A,0)*Ts); return PeriodicArray(M, A.period; nperiod) )

   K >= 100 ? dt = Ts : dt = Ts*K/100/nperiod

   Threads.@threads for i = 1:K
      @inbounds M[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver = solver, reltol = reltol, abstol = abstol, dt = dt) 
   end
   return PeriodicArray(M,A.period; nperiod)
end
"""
     pseig(A, K = 1; lifting = false, solver, reltol, abstol, dt) -> ev

Compute the characteristic multipliers of a continuous-time periodic matrix. 

For the given square periodic matrix `A(t)` of period `T`, 
the characteristic multipliers `ev` are the eigenvalues of 
the monodromy matrix `Ψ = Φ(T,0)`, where `Φ(t,τ)` is the state transition matrix satisfying the homogeneous linear ODE 

    dΦ(t,τ)/dt = A(t)Φ(t,τ),  Φ(τ,τ) = I. 

If `lifting = false`, `Ψ` is computed as a product of `K` state transition matrices 
`Ψ = Ψ_K*...*Ψ_1` (see [`monodromy`](@ref) with the associated keyword arguments). 
The eigenvalues are computed using the periodic Schur decomposition method of [1].

If `lifting = true`, `Ψ` is (implicitly) expressed as `Ψ = inv(N)*M`, where `M-λN` is a regular
pencil with `N` invertible and  
the eigenvalues of `M-λN` are the same as those of the matrix product
`Ψ := Ψ_K*...*Ψ_1`. 
An efficient version of the structure exploiting fast reduction method of [2] is employed, 
which embeds the determination of transition matrices into the reduction algorithm. 
This option may occasionally lead to inaccurate results for large values of `K`. 
`A` may be a [`PeriodicFunctionMatrix`](@ref), or a [`PeriodicSymbolicMatrix`](@ref), or a 
[`HarmonicArray`](@ref) or a [`PeriodicTimeSeriesMatrix`](@ref).

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.

"""
function pseig(at::PM, K::Int = 1; lifting::Bool = false, solver = "non-stiff", reltol = 1e-3, abstol = 1e-7, dt = at.period/100/at.nperiod) where 
   {T, PM <: Union{PeriodicFunctionMatrix{:c,T},HarmonicArray{:c,T},FourierFunctionMatrix{:c,T}}} 
   n = size(at,1)
   n == size(at,2) || error("the periodic matrix must be square")
   nperiod = at.nperiod
   t = 0  
   Ts = at.period/K/nperiod
   if lifting 
      if K == 1
         ev = eigvals(tvstm(at, at.period, 0; solver, reltol, abstol, dt)) 
      else   
         Z = zeros(T,n,n)
         ZI = [ Z; -I]
         si = tvstm(at, Ts, 0; solver, reltol, abstol); ti = -I
         t = Ts
         for i = 1:K-1
             tf = t+Ts
             F = qr([ ti; tvstm(at, tf, t; solver, reltol, abstol, dt) ])     
             si = F.Q'*[si; Z];  si = si[n+1:end,:]
             ti = F.Q'*ZI; ti = ti[n+1:end,:]
             t = tf
         end
         ev = -eigvals(si,ti)
      end
      sorteigvals!(ev)
   else
      M = monodromy(at, K; solver, reltol, abstol, dt) 
      ev = K == 1 ? eigvals(view(M.M,:,:,1)) : pschur(M.M; withZ = false)[3]
      isreal(ev) && (ev = real(ev))
   end
   return nperiod == 1 ? ev : ev.^nperiod
end
pseig(at::PeriodicSymbolicMatrix{:c,T}, K::Int = 1; kwargs...) where T = 
    pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)
# pseig(at::HarmonicArray{:c,T}, K::Int = 1; kwargs...) where T = 
#     pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)
pseig(at::PeriodicTimeSeriesMatrix{:c,T}, K::Int = 1; kwargs...) where T = 
    pseig(convert(PeriodicFunctionMatrix,at),K; kwargs...)
"""
     ev = pseig(A::PeriodicArray; rev = true, fast = false) 

Compute the eigenvalues of a product of `p` square matrices 
`A(p)...*A(2)*A(1)`, if `rev = true` (default) (also called characteristic multipliers) or 
of `A(1)*A(2)...A(p)` if `rev = false`, without evaluating the product. 
The matrices `A(1)`, `...`, `A(p)` are contained in the `n×n×p` array `A` 
such that the `i`-th matrix `A(i)` is contained in `A[:,:,i]`.
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
function pseig(A::PeriodicArray{:d,T}; fast::Bool = false) where T
   pseig(A.M; fast).^(A.nperiod)
end
function pseig(A::Array{T,3}; rev::Bool = true, fast::Bool = false) where T
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
      T1 = promote_type(Float64,T)
      ev = pschur(T1.(A); rev, withZ = false)[3]
      isreal(ev) && (ev = real(ev))
      return ev
   end
end
"""
     ev = pseig(A::PeriodicMatrix[, k = 1]; rev = true, fast = false) 

Compute the eigenvalues of a square cyclic product of `p` matrices 
`A(k-1)...*A(2)*A(1)*A(p)...*A(k)`, if `rev = true` (default) or 
`A(k)*A(k+1)*...A(p)*A(1)...A(k-1)` if `rev = false`, without evaluating the product. 
The argument `k` specifies the starting index (default: `k = 1`). 
The matrices `A(1)`, `...`, `A(p)` are contained in the `p`-vector of matrices `A` 
such that the `i`-th matrix  `A(i)`, of dimensions `m(i)×n(i)`, is contained in `A[i]`.
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
function pseig(A::PeriodicMatrix{:d,T}, k::Int = 1; fast::Bool = false) where T
   pseig(A.M, k; fast).^(A.nperiod)
end
function pseig(A::Vector{Matrix{T}}, k::Int = 1; rev::Bool = true, fast::Bool = false) where T
   p = length(A)
   istart = mod(k-1,p)+1
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
"""
     psceig(A::PeriodicFunctionMatrix[, K = 1]; lifting = false, solver, reltol, abstol, dt) -> ce

Compute the characteristic exponents of a periodic matrix.

For a given square continuous-time periodic function matrix `A(t)` of period `T`, 
the characteristic exponents `ce` are computed as `log.(ev)/T`, 
where  `ev` are the characteristic
multipliers (i.e., the eigenvalues of the monodromy matrix of `A(t)`).  
For available options see [`pseig(::PeriodicFunctionMatrix)`](@ref). 
For a given square discrete-time periodic matrix `A(t)` of discrete period `N`,  
the characteristic exponents `ce` are computed as `ev.^-N`. 
"""
function psceig(at::PM, K::Int = 1; kwargs...) where
   {T, PM <: Union{PeriodicFunctionMatrix{:c,T},HarmonicArray{:c,T},FourierFunctionMatrix{:c,T}}} 
   ce = log.(complex(pseig(at, K; kwargs...)))/at.period
   return isreal(ce) ? real(ce) : ce
end
function psceig(at::Union{PeriodicSymbolicMatrix, PeriodicTimeSeriesMatrix}, K::Int = 1; kwargs...) 
   ce = log.(complex(pseig(convert(PeriodicFunctionMatrix,at), K; kwargs...)))/at.period
   return isreal(ce) ? real(ce) : ce
end
"""
    psceighr(Ahr::HarmonicArray[, N]; P, nperiod, shift, atol) -> ce

Compute the characteristic exponents of a continuous-time periodic matrix in _Harmonic Array_ representation. 

For a given square continuous-time periodic function matrix `Ahr(t)` of period `T` 
in a  _Harmonic Array_ representation, 
the characteristic exponents `ce` are computed as the eigenvalues of a truncated harmonic state operator `A(N)-E(N)` lying in the 
fundamental strip `-ω/2 <  Im(λ) ≤ ω/2`, where `ω = 2π/T`. If `Ahr(t)` has the harmonic components `A_0`, `A_1`, ..., `A_p`, then 
for `N ≥ p`, `P = 1` and `nperiod = 1`, the matrices `A(N)` and `E(N)` are built as


           ( A_0  A_{-1} …  A_{-p}        0    )           ( -im*ϕ_{-N}I                                 0        )
           ( A_1   A_0             ⋱           )           (     ⋮       ⋱                                        )
           (  ⋮         ⋱            ⋱         )           (               -im*ϕ_{-1}*I                           )
    A(N) = ( A_p             ⋱          A_{-p} ) , E(N) =  (                           -im*ϕ_{0}*I                )
           (        ⋱           ⋱         ⋮    )           (     ⋮                                  ⋱              )
           (  0        A_p      …         A_0  )           (     0                                   -im*ϕ_{N}I   )

with `ϕ_{i} := shift+i*ω`. If `N < p`, then a truncated _full_ block Toeplitz matrix A(N) is built using the first `N` harmonic components. 
The default value used for `N` is `N = max(10,p-1)`. 
           
Generally, for given `P ≥ 1` and  `nperiod ≥ 1`, the block Toeplitz matrix `A(N)` (and also `E(N)`) is constructed with `(2N*np+1)×(2N*np+1)` blocks,
with `np = P*nperiod`, such that each `A_i` is preceeded in its column by `np-1` zero blocks, 
each `A_{-i}` is preceeded in its row by `np-1` zero blocks and all diagonal blocks are equal to`A_0`.  

The keyword argument `atol` (default: `atol = 1.e-10`) is a tolerance on the magnitude of the trailing components of the 
associated eigenvectors used to validate their asymptotic (exponential) decay. 
Only eigenvalues satisfying this check are returned in `ce`. 

_References_

[1] J. Zhou, T. Hagiwara, and M. Araki. 
    Spectral characteristics and eigenvalues computation of the harmonic state operators in continuous-time periodic systems. 
    Systems and Control Letters, 53:141–155, 2004.
"""
function psceighr(Ahr::HarmonicArray{:c,T}, N::Int = max(10,size(Ahr.values,3)-1); P::Int = 1, nperiod::Int = Ahr.nperiod, shift::Real = 0, atol::Real = 1.e-10) where T
   n = size(Ahr,1)
   n == size(Ahr,2) || error("the periodic matrix must be square") 
   (N == 0 || isconstant(Ahr)) && (return eigvals(real(Ahr.values[:,:,1])))
   ev, V = eigen!(hr2btupd(Ahr, N; P, shift, nperiod));
   ind = sortperm(imag(ev),by=abs); 
   ωhp2 = pi/P/Ahr.period/Ahr.nperiod*nperiod
   ne = count(abs.(imag(ev[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
   ev = ev[ind[1:ne]]
   # return only validated eigenvalues
   σ = Complex{T}[]
   for i = 1:ne
       norm(V[end-3:end,ind[i]]) < atol  && push!(σ,ev[i])
   end
   nv = length(σ)
   nv < n && @warn "number of eigenvalues is less than the order of matrix, try again with increased number of harmonics"
   ce = nv > n ? σ[sortperm(imag(σ),rev=true)][1:n] : σ[1:nv]
   return isreal(ce) ? real(ce) : ce
end
"""
    psceigfr(A::FourierFunctionMatrix[, N]; P, atol) -> ce

Compute the characteristic exponents of a continuous-time periodic matrix in _Fourier Function Matrix_ representation. 

For a given square continuous-time periodic function matrix `A(t)` of period `T` 
in a  _Fourier Function Matrix_ representation, 
the characteristic exponents `ce` are computed as the eigenvalues of the state operator `A(t)-D*I` lying in the 
fundamental strip `-ω/2 <  Im(λ) ≤ ω/2`, where `ω = 2π/T`. A finite dimensional truncated matrix of order `n*(2*N*P+1)` 
is built to approximate `A(t)-D*I`, where `n` is the order of `A(t)`,  `N` is the number of selected harmonic components
in the Fourier representation and `P` is the period multiplication number (default: `P = 1`).
The default value used for `N` is `N = max(10,p-1)`, where `p` the number of harmonics terms of `A(t)` (see [`FourierFunctionMatrix`](@ref)). 

The keyword argument `atol` (default: `atol = 1.e-10`) is a tolerance on the magnitude of the trailing components of the 
associated eigenvectors used to validate their asymptotic (exponential) decay. Only eigenvalues satisfying this check are returned in `ce`. 
"""
function psceigfr(Afun::FourierFunctionMatrix{:c,T}, N::Int = max(10,maximum(ncoefficients.(Matrix(Afun.M)))); P::Int = 1, atol::Real = 1.e-10) where T
   n = size(Afun,1)
   n == size(Afun,2) || error("the periodic matrix must be square") 
   (N == 0 || isconstant(Afun)) && (return eigvals(getindex.(coefficients.(Matrix(Afun.M)),1)))
   Af = P == 1 ? Afun :  FourierFunctionMatrix(Fun(t -> Afun.M(t),Fourier(0..P*Afun.period)))
   D = Derivative(domain(Af.M))

   Aop = Af.M - DiagDerOp(D,n)
   NA = n*(2*N*P+1)
   RW = Aop[1:NA,1:NA]
   W = Matrix(RW)
   ev, V = eigen(W)

   ind = sortperm(imag(ev),by=abs) 
   ωhp2 = pi/Af.period/Af.nperiod
   ne = count(abs.(imag(ev[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
   ev = ev[ind[1:ne]]
   # return only validated eigenvalues
   σ = Complex{T}[]
   for i = 1:ne
       norm(V[end-3:end,ind[i]]) < atol  && push!(σ,ev[i])
   end
   nv = length(σ)
   nv < n && @warn "number of eigenvalues is less than the order of matrix, try again with increased number of harmonics"
   ce = nv > n ? σ[sortperm(imag(σ),rev=true)][1:n] : σ[1:nv]
   return isreal(ce) ? real(ce) : ce
end
"""
    psceig(A::AbstractPeriodicArray[, k]; kwargs...) -> ce

Compute the characteristic exponents of a cyclic matrix product of `p` matrices.

The characteristic exponents of a product of `p` matrices are computed as the `p`th roots of the 
characteristic multipliers. These are computed as the eigenvalues of the square 
cyclic product of `p` matrices `A(k-1)...*A(2)*A(1)*A(p)...*A(k)`, if `rev = true` (default) or 
`A(k)*A(k+1)*...A(p)*A(1)...A(k-1)` if `rev = false`, without evaluating the product. 
The argument `k` specifies the starting index (default: `k = 1`). 
The matrices `A(1)`, `...`, `A(p)` are contained in the `p`-vector of matrices `A` 
such that the `i`-th matrix  `A(i)`, of dimensions `m(i)×n(i)`, is contained in `A[i]`.
The keyword arguments `kwargs` are those of  [`pseig(::PeriodicMatrix)`](@ref).  

_Note:_ The first `nmin` components of `ce` contains the _core characteristic exponents_ of the appropriate matrix product,
where `nmin` is the minimum row dimensions of matrices `A[i]`, for `i = 1, ..., p`, 
while the last components of `ce` are zero. 
"""
function psceig(at::AbstractPeriodicArray{:d,T}, k::Int = 1; kwargs...) where T
   ce = (complex(pseig(convert(PeriodicMatrix,at), k; kwargs...))).^(1/at.dperiod/at.nperiod) 
   return isreal(ce) ? real(ce) : ce
end

function sorteigvals!(ev)
   # an approximately complex conjugated set is assumed 
   isreal(ev) && (return ev)
   tc = ev[imag.(ev) .> 0]
   ev[:] = [ev[imag.(ev) .== 0]; sort([tc; conj.(tc)],by = real)]
   return ev
end

"""
     pschur(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur1(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

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
      return [St[1:mp[i],1:np[i],i] for i in 1:p], 
             withZ ? [Zt[1:mp[i],1:mp[i],i] for i in 1:p] : nothing, ev, sind, α, γ
   end
end
function pschur1(A::AbstractVector{Matrix{T}}; rev::Bool = true, withZ::Bool = true) where T

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
     pschur(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur1(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)
     pschur2(A; rev = true, withZ = true) -> (S, Z, ev, ischur, α, γ)

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
function pschur(A::Array{Float64,3}; rev::Bool = true, sind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions") 
   p = size(A,3)
   (sind < 1 || sind > p) && error("sind is out of range $(1:p)") 

   #  use ilo = 1 and ihi = n for the reduction to the periodic Hessenberg form
   ilo = 1; ihi = n; 
   #TAU = Array{Float64,2}(undef, max(n-1,1), p)
   
   if withZ 
      Z = Array{Float64,3}(undef, n, n, p) 
      compQ = 'I'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   if rev
      imap = mod.(Vector(sind+p-1:-1:sind),p) .+ 1
      S = A[:,:,imap]
      hc = 1
   else
      S = copy(A)
      hc = sind  
   end

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*p,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   SIND = ones(BlasInt,p)

   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, p, hc, ilo, ihi, SIND, S, Z, LIWORK, LDWORK)

   # reduce to periodic Schur form
   LDWORK = p + max( 2*n, 8*p )
   LIWORK = 2*p + n
   ALPHAR = Array{Float64,1}(undef, n)
   ALPHAI = Array{Float64,1}(undef, n)
   BETA = Array{Float64,1}(undef, n)
   SCAL = Array{BlasInt,1}(undef, n)
   withZ && (compQ = 'U')
   SLICOTtools.mb03bd!('T','C',compQ, QIND, p, n, hc, ilo, ihi, SIND, S, Z, ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)
   α = complex.(ALPHAR, ALPHAI)
   γ = 2. .^SCAL
   ev = α .* γ

   if rev
      return S[:,:,imap], withZ ? Z[:,:,mod.(p+sind:-1:sind+1,p).+1] : nothing, ev, sind, α, γ
   else
      return S, withZ ? Z : nothing, ev, sind, α, γ
   end
end
"""
     psordschur!(S, Z, select; rev, sind) 

Reorder the eigenvalues of the product `Π = S[p]*...*S[2]*S[1]`, if `rev = true` (default) or `Π = S[1]*S[2]*...*S[p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The `p`-vectors `S` and `Z` contain the matrices `S[1]`, `...`, `S[p]` in a periodic Schur form, with `S[sind]` in real Schur form, 
and the corresponding orthogonal transformation matrices `Z[1]`, `...`, `Z[p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function psordschur!(S::AbstractVector{Matrix{T}}, Z::AbstractVector{Matrix{T}}, select::Union{Vector{Bool},BitVector}; rev::Bool = true, sind::Int = 1) where T
   k = length(S) 
   m = size.(S,1); n = size.(S,2) 
   nc = n[1]; 
   (all(m .== nc) && all(n .== nc)) || error("all elements of S must be square matrices of same dimension")
   k == length(Z) || error("S and Z must have the same length")
   (all(size.(Z,1) .== nc) && all(size.(Z,2) .== nc)) || error("all elements of Z must be square matrices of same dimension with S")
   kschur = rev ? sind : k-sind+1
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
     psordschur!(S, Z, select; rev, sind) 

Reorder the eigenvalues of the product `Π = S[:,:,p]*...*S[:,:,2]*S[:,:,1]`, if `rev = true` (default) or `Π = S[:,:,1]*S[:,:,2]*...*S[:,:,p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The 3-dimensional arrays `S` and `Z` contain the matrices `S[:,:,1]`, `...`, `S[:,:,p]` in a periodic Schur form, with `S[:,:,sind]` in real Schur form, 
and the corresponding orthogonal transformation matrices `Z[:,:,1]`, `...`, `Z[:,:,p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.    
"""
function psordschur!(S::Array{T,3}, Z::Array{T,3}, select::Union{Vector{Bool},BitVector}; rev::Bool = true, sind::Int = 1) where T

   mc, nc, k = size(S) 
   mc == nc || error("S must have the same first and second dimensions")
   mz, nz, kz = size(Z) 
   k == kz || error("S and Z must have the same length")
   mz == nz == nc || error("Z must have the same first and second dimensions as S")
   kschur = rev ? sind : k-sind+1
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
     psordschur1!(S, Z, select; rev, sind) 

Reorder the core eigenvalues of the product `Π = S[p]*...*S[2]*S[1]`, if `rev = true` (default) or `Π = S[1]*S[2]*...*S[p]`
if `rev = false`, where `Π` is in real Schur form, such that the selected eigenvalues in the logical array `select` are moved into the leading positions. 
The `p`-vectors `S` and `Z` contain the matrices `S[1]`, `...`, `S[p]` in an extended periodic Schur form, with the leading square block of 
`S[sind]` in real Schur form, and the corresponding orthogonal transformation matrices `Z[1]`, `...`, `Z[p]`, respectively.  `S` and `Z` are overwritten by the updated matrices. 
A conjugate pair of eigenvalues must be either both included or both excluded via `select`.  The dimension of select must be equal to the number of
core eigenvalues (i.e., the minimum dimension of matrices in the vector `S`).  
"""
function psordschur1!(S::AbstractVector{Matrix{T}}, Z::AbstractVector{Matrix{T}}, select::Union{Vector{Bool},BitVector}; rev::Bool = true, sind::Int = 1) where T

   k = length(S) 
   m = size.(S,1); n = size.(S,2) 
   rev || (reverse!(m); reverse!(n))
   ldq = rev ? n : [m[end];m[1:end-1]]
   nc = minimum(n)
   kschur = rev ? sind : k-sind+1
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
   α = complex.(ALPHAR, ALPHAI)
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
     phess(A; hind = 1, rev = true, withZ = true) -> (H, Z, ihess)
     phess1(A; hind = 1, rev = true, withZ = true) -> (H, Z, ihess)

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
function phess(A::AbstractArray{Float64,3}; rev::Bool = true, hind::Int = 1, withZ::Bool = true)
   n = size(A,1)
   n == size(A,2) || error("A must have equal first and second dimensions")
   K = size(A,3)
   (hind < 1 || hind > K) && error("hind is out of range $(1:K)") 

   ilo = 1; ihi = n; 
   if withZ 
      Z = Array{Float64,3}(undef, n, n, K) 
      compQ = 'I'
   else
      Z = Array{Float64,3}(undef, 0, 0, 0)
      compQ = 'N'
   end

   if rev
      imap = mod.(Vector(hind+K-1:-1:hind),K) .+ 1
      H = A[:,:,imap]
      hc = 1
   else
      H = copy(A)
      hc = hind  
   end

   # reduce to periodic Hessenberg form
   LDWORK = max(2*n,1)
   LIWORK = max(3*K,1)
   QIND = Array{BlasInt,1}(undef, 1) 
   
   SLICOTtools.mb03vw!(compQ, QIND, 'A', n, K, hc, ilo, ihi, ones(BlasInt,K), H, Z, LIWORK, LDWORK)

   if rev
      return H[:,:,imap], withZ ? Z[:,:,mod.(K+hind:-1:hind+1,K).+1] : nothing, hind
   else
      return H, withZ ? Z : nothing, hind
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
as provided in the [`Interpolations.jl`](https://github.com/JuliaMath/Interpolations.jl)  package. 
The keyword parameter `method` specifies the interpolation/extrapolation method to be used as follows:

`method = "constant"` - use periodic B-splines of degree 0 (periodic constant interpolation);

`method = "linear"` - use periodic B-splines of degree 1 (periodic linear interpolation) (default);

`method = "quadratic"` - use periodic B-splines of degree 2 (periodic quadratic interpolation); 

`method = "cubic"` - use periodic B-splines of degree 3 (periodic cubic interpolation). 
"""
function ts2pfm(A::PeriodicTimeSeriesMatrix; method = "linear")
   N = length(A.values)
   N == 0 && error("empty time array not supported")
   N == 1 && (return PeriodicFunctionMatrix(t -> A.values[1], A.period; nperiod = A.nperiod, isconst = true))
   #ts = (0:N-1)*(A.period/N)
   ts = (0:N-1)*(A.period/N/A.nperiod)
   n1, n2 = size(A.values[1])
   intparray = Array{Any,2}(undef, n1, n2)
   if method == "constant"      
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Constant(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "linear" || N == 2
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Linear(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "quadratic" || N == 3     
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Quadratic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "cubic"     
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Cubic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   else
      error("no such option method = $method")
   end
   return PeriodicFunctionMatrix(t -> [intparray[i,j](t) for i in 1:n1, j in 1:n2 ], A.period; nperiod = A.nperiod, isconst = isconstant(A))
end
"""
     ts2hr(A::PeriodicTimeSeriesMatrix; atol = 0, rtol, n, squeeze = true) -> Ahr::HarmonicArray

Compute the harmonic (Fourier) approximation of a periodic matrix specified by a time series matrix. 
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
are `atol = 0` and `rtol = 10*p*ϵ`, where `ϵ` is the working machine precision.

The resulting harmonic approximation `Ahr(t)` is returned in the harmonic array object `Ahr` 
(see [`HarmonicArray`](@ref)). 
"""
function ts2hr(A::PeriodicTimeSeriesMatrix{:c,T}; atol::Real = 0, rtol::Real = 0, n::Union{Int,Missing} = missing, squeeze::Bool = true) where  {T}
   
   M = length(A.values)
   n1, n2 = size(A.values[1])
   
   if ismissing(n)
       n = div(M-1,2)
       ncur = 0
   else
       n = min( n, div(M-1,2) ) 
       ncur = n
   end
   n = max(n,0)
   
   AHR = zeros(ComplexF64, n1, n2, n+1)
   T1 = promote_type(Float64,T)
   tol = iszero(atol) ? (iszero(rtol) ? 10*M*eps(T1)*maximum(norm.(A.values)) : rtol*maximum(norm.(A.values)) ) : atol
   i1 = 1:n+1   
   for i = 1:n1
       for j = 1:n2
           temp = T1.(getindex.(A.values, i, j))
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
   nperiod0 = A.nperiod
   nperiod = 1
   if ncur > 2 && squeeze
      nh = ncur-1
      s = falses(nh)
      for i = 1:nh
          s[i] = any(abs.(view(AHR,:,:,i+1)) .> tol)
      end  
      t = Primes.factor(Vector,nh)
      s1 = copy(s)
     for i = 1:length(t)
          stry = true
          for j1 = 1:t[i]:nh
              stry = stry & all(view(s1,j1:j1+t[i]-2) .== false) 
              stry || break
          end
          if stry 
             nperiod = nperiod*t[i]
             s1 = s1[t[i]:t[i]:nh]
             nh = div(nh,t[i])
          end
      end 
      return HarmonicArray(AHR[:,:,[1;nperiod+1:nperiod:ncur]],A.period;nperiod = nperiod*nperiod0)
   else
      return HarmonicArray(AHR[:,:,1:max(1,ncur)],A.period;nperiod = nperiod0)
   end       

end
"""
     ffm2hr(Afun::FourierFunctionMatrix; atol = 0, rtol = √ϵ, squeeze = true) -> Ahr::HarmonicArray

Compute the harmonic (Fourier) representation of a Fourier periodic matrix object. 

The Fourier function matrix object `Afun` of period `T` is built using
the Fourier series representation of a periodic matrix `Afun(t)` of subperiod `T′ = T/k`, 
where each entry of `Afun(t)` has the form

             p
      a_0 +  ∑ ( ac_i*cos(i*t*2*π/T′)+as_i*sin(i*2*π*t/T′) ) ,
            i=1 

where `k ≥ 1` is the number of subperiods (default: `k = 1`).   

The harmonic array object `Ahr` of period `T` is built using
the harmonic representation of a periodic matrix `Ahr(t)` of subperiod `T′′ = T/k′` in the form

                     p′
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T′′)+As_i*sin(i*2*π*t/T′′) ) ,
                    i=1 

where `p′` is the maximum index `j`, such that `Ac_j` and/or `As_j` are nonzero.
The tolerance used to assess nonzero elements is `tol = max(atol, rtol*maxnorm)`, where 
`maxnorm` is the maximum absolute value of the coefficients `ac_i` and `as_i` in `Afun(t)`. The default values of tolerances
are `atol = 0` and `rtol = √ϵ`, where `ϵ` is the working machine precision.
The resulting harmonic approximation `Ahr(t)` is returned in the harmonic array object `Ahr` 
(see [`HarmonicArray`](@ref)). 
"""
function ffm2hr(A::FourierFunctionMatrix{:c,T}; atol::Real = 0, rtol::Real = 0, squeeze::Bool = true) where  {T}
   lens = length.(coefficients.(Matrix(A.M)))
   n = max(div(maximum(lens)-1,2),0)
   n1, n2 = size(A)

   ncur = n
   AHR = zeros(complex(T), n1, n2, n+1)
   tol = iszero(atol) ? (iszero(rtol) ? 10*n*eps(maximum(norm.(coefficients.(Matrix(A.M)),Inf))) : rtol*maximum(norm.(coefficients.(Matrix(A.M)),Inf)) ) : atol
   for i = 1:n1
       for j = 1:n2
           tt = coefficients(A.M[i,j])
           tt[abs.(tt) .<= tol] .= zero(T)
           lentt = lens[i,j]
           if lentt > 0
              AHR[i,j,1] = tt[1] 
              k = 1
              for k1 = 2:2:lentt-1
                  k += 1
                  AHR[i,j,k] = tt[k1+1] + im*tt[k1]
              end
              isodd(lentt) || (AHR[i,j,k+1] = im*tt[end])
           end
       end
   end
   nperiod = A.nperiod
   if ncur > 2 && squeeze
      nh = ncur-1
      s = falses(nh)
      for i = 1:nh
          s[i] = any(abs.(view(AHR,:,:,i+1)) .> tol)
      end  
      t = Primes.factor(Vector,nh)
      s1 = copy(s)
      for i = 1:length(t)
          stry = true
          for j1 = 1:t[i]:nh
              stry = stry & all(view(s1,j1:j1+t[i]-2) .== false) 
              stry || break
          end
          if stry 
             nperiod = nperiod*t[i]
             s1 = s1[t[i]:t[i]:nh]
             nh = div(nh,t[i])
          end
      end 
      return HarmonicArray(AHR[:,:,[1;nperiod+1:nperiod:ncur]],A.period;nperiod)
   else
      return HarmonicArray(AHR[:,:,1:max(1,ncur+1)],A.period;nperiod)
   end       

end

# function ts2ffm(A::PeriodicTimeSeriesMatrix{:c,T}; atol::Real = 0, rtol::Real = 0, n::Union{Int,Missing} = missing, squeeze::Bool = true) where  {T, Ts}
   
#    M = length(A.values)
#    n1, n2 = size(A.values[1])
   
#    if ismissing(n)
#        n = div(M-1,2)
#        ncur = 0
#    else
#        n = min( n, div(M-1,2) ) 
#        ncur = n
#    end
#    n = max(n,0)
#    s = Fourier(0..A.period)
#    AFFM = Fun(t -> zeros(T,n1,n2),s)
#    tol = iszero(atol) ? (iszero(rtol) ? 10*M*eps(maximum(norm.(A.values))) : rtol*maximum(norm.(A.values)) ) : atol
#    i1 = 2:n+1   
#    for i = 1:n1
#        for j = 1:n2
#            temp = getindex.(A.values, i, j)
#            i == 1 && j == 1 && (global rfftop = plan_rfft(temp; flags=FFTW.ESTIMATE, timelimit=Inf))
#            tt = conj(2/M*(rfftop*temp)) 
#          #   tt[1] = real(tt[1]/2)
#          #   tt1 = view(tt,i1)
#            ind = i1[abs.(view(tt,i1)) .> tol]
#            nc = length(ind); 
#            nc > 0 && (nc = ind[end])
#          #   nr = count(abs.(real(tt1)) .> tol)
#          #   indi = i1[abs.(imag(tt1)) .> tol]
#          #   ni = length(indi); 
#          #   ni > 0 && (ni = indi[end])
#          #   nc = count(view(tt))        
#          #   AFFM[i,j] = Fun(s,real(tt[indr])
#          #   AHR[i,j,indi] += im*imag(tt[indi])
#            i2 = 2:nc
#            AFFM[i,j] = Fun(s, [[real(tt[1])/2]; transpose([real(tt[i2]) imag(tt[i2])])[:]])
#        end
#    end
#    # nperiod = A.nperiod
#    # if nc > 2 && squeeze
#    #    nh = ncur-1
#    #    s = falses(nh)
#    #    for i = 1:nh
#    #        s[i] = any(abs.(view(AHR,:,:,i+1)) .> tol)
#    #    end  
#    #    t = Primes.factor(Vector,nh)
#    #    s1 = copy(s)
#    #    for i = 1:length(t)
#    #        #st = falses(0) 

#    #        #k = div(nh,t[i])  # number of possible patterns
#    #        stry = true
#    #        for j1 = 1:t[i]:nh
#    #            stry = stry & all(view(s1,j1:j1+t[i]-2) .== false) 
#    #            stry || break
#    #            #st = [st; s1[j1+t[i]-1]]
#    #        end
#    #        if stry 
#    #           nperiod = nperiod*t[i]
#    #           s1 = s1[t[i]:t[i]:nh]
#    #           nh = div(nh,t[i])
#    #        end
#    #    end 
#    return FourierFunctionMatrix(AFFM,A.period,A.nperiod)
#    # else
#    #    return FourierFunctionMatrix(AHR[:,:,1:max(1,ncur)],A.period;nperiod)
#    # end       

# end
"""
     pfm2hr(A::PeriodicFunctionMatrix; nsample, NyquistFreq) -> Ahr::HarmonicArray

Convert a periodic function matrix into a harmonic array representation. 
If `A(t)` is a periodic function matrix of period `T`, then the harmonic array representation
`Ahr` is determined by applying the fast Fourier transform to the sampled arrays `A(iΔ)`, `i = 0, ..., k`,
where `Δ = T/k` is the sampling period and `k` is the number of samples specified by the keyword argument
`nsample = k` (default: `k = 128`). If the Nyquist frequency `f` is specified via the keyword argument
`NyquistFreq = f`, then `k` is chosen `k = 2*f*T` to avoid signal aliasing.     
"""
function pfm2hr(A::PeriodicFunctionMatrix; nsample::Int = 128, NyquistFreq::Union{Real,Missing} = missing)   
   isconstant(A) && (return HarmonicArray(A.f(0),A.period; nperiod = A.nperiod))
   nsample > 0 || ArgumentError("nsample must be positive, got $nsaple")
   ns = ismissing(NyquistFreq) ? nsample : Int(floor(2*abs(NyquistFreq)*A.period/A.nperiod))+1
   Δ = A.period/ns/A.nperiod
   ts = (0:ns-1)*Δ
   return ts2hr(PeriodicTimeSeriesMatrix(A.f.(ts), A.period; nperiod = A.nperiod),squeeze = true)
end
"""
     psm2hr(A::PeriodicSymbolicMatrix; nsample, NyquistFreq) -> Ahr::HarmonicArray

Convert a periodic symbolic matrix into a harmonic array representation. 
If `A(t)` is a periodic symbolic matrix of period `T`, then the harmonic array representation
`Ahr` is determined by applying the fast Fourier transform to the sampled arrays `A(iΔ)`, `i = 0, ..., k`,
where `Δ = T/k` is the sampling period and `k` is the number of samples specified by the keyword argument
`nsample = k` (default: `k = 128`). If the Nyquist frequency `f` is specified via the keyword argument
`NyquistFreq = f`, then `k` is chosen `k = 2*f*T` to avoid signal aliasing.     
"""
function psm2hr(A::PeriodicSymbolicMatrix; nsample::Int = 128, NyquistFreq::Union{Real,Missing} = missing)   
   isconstant(A) && (return HarmonicArray(convert(PeriodicFunctionMatrix,A).f(0),A.period; nperiod = A.nperiod))
   nsample > 0 || ArgumentError("nsample must be positive, got $nsaple")
   ns = ismissing(NyquistFreq) ? nsample : Int(floor(2*abs(NyquistFreq)*A.period))+1
   Δ = A.period/ns
   ts = (0:ns-1)*Δ
   return ts2hr(PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,A).f.(ts), A.period; nperiod = A.nperiod))
end
"""
     hrchop(Ahr::HarmonicArray; tol) -> Ahrtrunc::HarmonicArray

Remove the trailing terms of a harmonic representation by deleting those whose norms are below a certain tolerance. 
"""
function hrchop(ahr::HarmonicArray; tol::Real = sqrt(eps()) ) 
   nrm = norm(ahr.values[:,:,1],1)
   n3 = size(ahr.values,3)
   for i = 2:n3
       nrm = max(nrm,norm(ahr.values[:,:,i],1))
   end
   itrunc = 1
   atol = tol*nrm
   for i = n3:-1:1
       @show 
       if norm(ahr.values[:,:,i],1) > atol 
          itrunc = i
          break
       end
   end
   return HarmonicArray(ahr.values[:,:,1:itrunc], ahr.period; nperiod = ahr.nperiod)
end
"""
     hrtrunc(Ahr::HarmonicArray, n) -> Ahrtrunc::HarmonicArray

Truncate a harmonic representation by deleting the trailing terms whose indices exceed certain number `n` of harmonics. 

"""
function hrtrunc(ahr::HarmonicArray, n::Int = size(ahr.values,3)-1) 
   return HarmonicArray(ahr.values[:,:,1:max(1,min(n+1,size(ahr.values,3)))], ahr.period; nperiod = ahr.nperiod)
end

"""
     hr2psm(Ahr::HarmonicArray, nrange) -> A::Matrix{Num}

Convert a range of harmonic components `nrange` of the harmonic array `Ahr` to a symbolic matrix `A`. 
The default range is `nrange = 0:n`, where `n` is the order of the maximum harmonics.  
"""
function hr2psm(ahr::HarmonicArray, nrange::UnitRange = 0:size(ahr.values,3)-1)   
   Symbolics.@variables t   
   Period = ahr.period
   i1 = max(first(nrange),0)
   i2 = min(last(nrange), size(ahr.values,3)-1)
   
   ts = t*2*pi*ahr.nperiod/Period
   
   i1 == 0 ? a = Num.(real.(ahr.values[:,:,1])) : (i1 -=1; a = zeros(Num,size(ahr.values,1),size(ahr.values,2)))
   for i in i1+1:i2
       ta = view(ahr.values,:,:,i+1)
       a .+= real.(ta).*(cos(i*ts)) .+ imag.(ta) .* (sin(i*ts))
   end
   return a
end 
"""
     pm2pa(At::PeriodicMatrix) -> A::PeriodicArray

Convert a discrete-time periodic matrix object into a discrete-time periodic array object.

The discrete-time periodic matrix object `At` contains a  
`p`-vector `At` of real matrices `At[i]`, `i = 1,..., p`, 
the associated time period `T` and the number of subperiods `k`. The resulting discrete-time periodic array object
`A` of period `T` and number of subperiods `k` 
is built from a `m×n×p` real array `A`, such that `A[:,:,i]` 
contains `At[i]`, for `i = 1,..., p`. 
For non-constant dimensions, the resulting `m` and `n` are the maximum row and column dimensions, respectively,
and the resulting component matrices `A[:,:,i]` contain `At[i]`, appropriately padded with zero entries. 
"""
function pm2pa(A::PeriodicMatrix{:d,T}) where T
   N = length(A)
   m, n = size(A)
   N == 0 && PeriodicArray(Array{T,3}(undef,0,0,0),A.period)
   if any(m .!= m[1]) || any(m .!= m[1]) 
      #@warn "Non-constant dimensions: the resulting component matrices padded with zeros"
      t = zeros(T,maximum(m),maximum(n),N)
      [copyto!(view(t,1:m[i],1:n[i],i),A.M[i]) for i in 1:N]
      PeriodicArray{:d,T}(t,A.period,A.nperiod)
   else
      t = zeros(T,m[1],n[1],N)
      [copyto!(view(t,:,:,i),A.M[i]) for i in 1:N]
      PeriodicArray{:d,T}(t,A.period,A.nperiod)
   end
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
The structure exploiting fast reduction method of [1] is employed to determine `M` and `N`.

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
       Z = zeros(T,n,n)
       ZI = [Z; -I]
       si = A[:,:,1];  ti = -I
       for i = 1:K-1
           F = qr([ ti; A[:,:,i+1] ])     
           si = F.Q'*[si; Z];  si = si[n+1:end,:]
           ti = F.Q'*ZI; ti = ti[n+1:end,:]
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
The structure exploiting fast reduction method of [1] is employed to determine `M` and `N`.

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

Evaluate the time response of a periodic time series matrix.

For the periodic time series matrix `At` and the vector of time values `t`, 
an interpolation/extrapolation based approximation  
`A[i]` is evaluated for each time value `t[i]`. The keyword parameter `method` specifies the
interpolation/extrapolation method to be used for periodic data. 
The following interpolation methods from the [`Interpolations.jl`](https://github.com/JuliaMath/Interpolations.jl) 
package can be selected: 

`method = "constant"` - use periodic B-splines of degree 0; 

`method = "linear"` - use periodic B-splines of degree 1 (periodic linear interpolation) (default);

`method = "quadratic"` - use periodic B-splines of degree 2 (periodic quadratic interpolation); 

`method = "cubic"` - use periodic B-splines of degree 3 (periodic cubic interpolation).
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
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Linear(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "cubic"      
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Cubic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "quadratic"      
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Quadratic(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   elseif method == "constant"      
      [intparray[i,j] = scale(Interpolations.extrapolate(interpolate(getindex.(A.values,i,j), BSpline(Constant(Periodic(OnCell())))), Periodic()), ts) for i in 1:n1, j in 1:n2]
   else
      error("no such option method = $method")
   end
   return [[intparray[i,j].(te[k]) for i in 1:n1, j in 1:n2 ] for k = 1:nt ]
end
"""
     hreval(Ahr::HarmonicArray, t; ntrunc, exact = true) -> A::Matrix

Evaluate the harmonic array `Ahr` representing a continuous-time 
time periodic matrix `A(t)` for a numerical or symbolic time value `t`. 
For a real value `t`, if `exact = true (default)` an exact evaluation is computed, while for `exact = false`, 
a linear interpolation based approximation is computed (potentially more accurate in intersample points).
The keyword argument `ntrunc` specifies the number of harmonics to be used for the evaluation 
(default: maximum possible number). 
If `t` is a symbolic variable, a symbolic evaluation of `A(t)` is performed (see also [`hr2psm`](@ref))
"""
function hreval(ahr::HarmonicArray{:c,T}, t::Union{Num,Real}; exact::Bool = true, ntrunc::Int = max(size(ahr.values,3)-1,0)) where {T}
      (ntrunc < 0 || ntrunc >= size(ahr.values,3)) && error("ntrunc out of allowed range")
   isa(t,Num) && (return hr2psm(ahr, 0:ntrunc))

   n = ntrunc   
   T1 = float(promote_type(T,typeof(t)))
   ts = mod(T1(t),T1(ahr.period))*2*pi*ahr.nperiod/ahr.period
   
   # determine interpolation coefficients
   ht = ones(T1,n);
   if !exact
      # use linear interpolation
      for i = 2:n
           x = pi*(i-1)/n
           ht[i] = (sin(x)/x)^2
      end
   end
   a = T == T1 ? real.(ahr.values[:,:,1]) : T1.(real.(ahr.values[:,:,1]))
   for i = 1:n
       ta = view(ahr.values,:,:,i+1)
       a .+= T1.(real.(ta)).*(cos(i*ts)*ht[i]) .+ T1.(imag.(ta)) .* ((sin(i*ts)*ht[i]))
   end
   return a
end   
"""
     tvmeval(Ahr::HarmonicArray, t; ntrunc, exact = true) -> A::Vector{Matrix}

Evaluate the time response of a harmonic array.

For the harmonic array `Ahr` representing representing a continuous-time 
time periodic matrix `A(t)` and the vector of time values `t`, 
`A[i] = A(t[i])` is computed for each time value `t[i]`. 
If `exact = true (default)` an exact evaluation is computed, while for `exact = false`, 
a linear interpolation based approximation is computed 
(potentially more accurate in intersample points).
The keyword argument `ntrunc` specifies the number of harmonics to be used for evaluation 
(default: maximum possible number of harmonics). 
"""
function tvmeval(ahr::HarmonicArray{:c,T}, t::Union{Real,Vector{<:Real}}; ntrunc::Int = size(ahr.values,3), 
                exact::Bool = true) where {T}
       
   n = min(size(ahr.values,3)-1,ntrunc);
   
   isa(t,Real) ? te = [t] : te = t
   nt = length(te)
   period = ahr.period
   
   tscal = 2*pi*ahr.nperiod/period
   # determine interpolation coefficients
   ht = ones(Float64,n);
   if !exact
      # use linear interpolation
      for i = 2:n
           x = pi*(i-1)/n
           ht[i] = (sin(x)/x)^2
      end
   end
   T1 = float(T)
   A = similar(Vector{Matrix{T1}}, nt)
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

Evaluate the time response of a periodic symbolic matrix.

For the periodic symbolic matrix `Asym` representing a time periodic matrix `A(t)`
and the vector of time values `t`, 
`A[i] = A(t[i])` is evaluated for each time value `t[i]`. 
"""
function tvmeval(A::PeriodicSymbolicMatrix, t::Union{Real,Vector{<:Real}} )
   te = isa(t,Real) ? [mod(t,A.period)] : mod.(t,A.period)
   return (convert(PeriodicFunctionMatrix,A).f).(te)
end
"""
     tvmeval(Af::PeriodicFunctionMatrix, t) -> A::Vector{Matrix}

Evaluate the time response of a periodic function matrix.

For the periodic function matrix `Af` representing a time periodic matrix `A(t)` and the vector of time values `t`, 
`A[i] = A(t[i])` is evaluated for each time value `t[i]`. 
"""
function tvmeval(A::PeriodicFunctionMatrix, t::Union{Real,Vector{<:Real}} )
   te = isa(t,Real) ? [mod(t,A.period)] : mod.(t,A.period)
   return (A.f).(te)
end
"""
     tvmeval(A::FourierFunctionMatrix, t) -> Aval::Vector{Matrix}

Evaluate the time response of a periodic function matrix.

For the periodic matrix `A(t)`, in a Fourier Function Matrix representation, and the vector of time values `t`, 
`Aval[i] = A(t[i])` is evaluated for each time value `t[i]`. 
"""
function tvmeval(A::FourierFunctionMatrix, t::Union{Real,Vector{<:Real}} )
   te = isa(t,Real) ? [mod(t,A.period)] : mod.(t,A.period)
   return (A.M).(te)
end
function tpmeval(A::PeriodicFunctionMatrix, t::Real )
"""
     tpmeval(A, tval) -> Aval::Matrix

Evaluate the time value of a continuous-time periodic matrix.

For the periodic matrix `A(t)` and the time value `tval`, `Aval = A(tval)` is evaluated for the time value `t = tval`. 
"""
   return (A.f).(t)
end
tpmeval(A::FourierFunctionMatrix, t::Real ) = (A.M)(t)
tpmeval(A::HarmonicArray, t::Real ) = hreval(A, t; exact = true) 

"""
    pmaverage(A) -> Am 

Compute for the continuous-time periodic matrix `A(t)` 
the corresponding time averaged matrix `Am` over one period.  
"""
function pmaverage(A::PM) where {PM <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,PeriodicTimeSeriesMatrix}} 
   return real(convert(HarmonicArray,A).values[:,:,1])
end
pmaverage(A::HarmonicArray) = real(A.values[:,:,1])
pmaverage(A::FourierFunctionMatrix) = getindex.(coefficients.(Matrix(A.M)),1)

function getpm(A::PeriodicMatrix, k, dperiod::Union{Int,Missing} = missing)
   i = ismissing(dperiod) ? mod(k-1,A.dperiod)+1 : mod(k-1,dperiod)+1
   return A.M[i]
   #return view(A.M,i)
end
function getpm(A::PeriodicArray, k, dperiod::Union{Int,Missing} = missing)
   i = ismissing(dperiod) ? mod(k-1,A.dperiod)+1 : mod(k-1,dperiod)+1
   return A.M[:,:,i]
   #return view(A.M,:,:,i)
end
function copypm!(Dest::AbstractMatrix{T}, A::PeriodicMatrix{:d,T}, k, dperiod::Union{Int,Missing} = missing) where {T}
   i = ismissing(dperiod) ? mod(k-1,A.dperiod)+1 : mod(k-1,dperiod)+1
   return copyto!(Dest,view(A.M[i],:,:))
   #return copyto!(Dest,A.M[i])
   #return view(A.M,i)
end
function copypm!(Dest::AbstractMatrix, A::PeriodicArray, k, dperiod::Union{Int,Missing} = missing)
   i = ismissing(dperiod) ? mod(k-1,A.dperiod)+1 : mod(k-1,dperiod)+1
   return copyto!(Dest,view(A.M,:,:,i))
   #return view(A.M,:,:,i)
end

