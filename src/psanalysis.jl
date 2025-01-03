"""
    pspole(psys::PeriodicStateSpace{PM}[,K]; kwargs...) -> val

Return for the periodic system `psys = (A(t),B(t),C(t),D(t))` the complex vector `val` containing 
the characteristic exponents of the periodic matrix `A(t)` (also called the _poles_ of the system `psys`). 

Depending on the underlying periodic matrix type `PM`, the optional argument `K` and keyword arguments `kwargs` may have the following values:

- if `PM = PeriodicFunctionMatrix`, or `PM = PeriodicSymbolicMatrix`, or `PM = PeriodicTimeSeriesMatrix`, then `K` is the number of factors used to express the monodromy matrix of `A(t)` (default: `K = 1`)  and `kwargs` are the keyword arguments of  `PeriodicMatrices.pseig(::PeriodicFunctionMatrix)`; 

- if `PM = HarmonicArray`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  `PeriodicMatrices.psceig(::HarmonicArray)`; 

- if `PM = FourierFunctionMatrix`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  `PeriodicMatrices.psceig(::FourierFunctionMatrix)`; 

- if `PM = PeriodicMatrix` or `PM = PeriodicArray`, then `K` is the starting sample time (default: `K = 1`)  and `kwargs` are the keyword arguments of  `PeriodicMatrices.psceig(::PeriodicMatrix)`; 
"""
pspole(psys::PeriodicStateSpace{<: PeriodicArray}, N::Int = 1; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicMatrix}, N::Int = 1; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicFunctionMatrix}, N::Int = 10; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicTimeSeriesMatrix}, N::Int = 10; method = "cubic", kwargs...) = psceig(psys.A, N; method, kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicSwitchingMatrix}) = psceig(psys.A)
pspole(psys::PeriodicStateSpace{<: SwitchingPeriodicMatrix}) = psceig(convert(PeriodicMatrix,psys.A))
pspole(psys::PeriodicStateSpace{<: HarmonicArray}, N::Int = 10; kwargs...) = psceighr(psys.A, N; kwargs...)
#pspole(psys::PeriodicStateSpace{<: HarmonicArray}, N::Int = 10; kwargs...) = psceig(psys.A, N; kwargs...)  # fallback version
#pspole(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, N::Int = 10; kwargs...)  = psceigfr(psys.A, N; kwargs...)
"""
    pszero(psys::PeriodicStateSpace{HarmonicArray}[, N]; P, atol, rtol, fast) -> val
    pszero(psys::PeriodicStateSpace{PeriodicFunctionMatrix}[, N]; P, atol, rtol, fast) -> val
    pszero(psys::PeriodicStateSpace{PeriodicSymbolicMatrix}[, N]; P, atol, rtol, fast) -> val

Compute the finite and infinite zeros of a continuous-time periodic system `psys = (A(t), B(t), C(t), D(t))` in `val`, 
where the periodic system matrices `A(t)`, `B(t)`, `C(t)`, and `D(t)` are in a _Harmonic Array_, or 
_Periodic Function Matrix_, or _Periodic Symbolic Matrix_ representation 
(the last two representation are automatically converted to a _Harmonic Array_ representation). 
`N` is the number of selected harmonic components in the Fourier series of the system matrices (default: `N = max(20,nh-1)`, 
where `nh` is the maximum number of harmonics terms) and the keyword parameter `P` is the number of full periods 
to be considered (default: `P = 1`) to build 
a frequency-lifted LTI representation based on truncated block Toeplitz matrices. 

The computation of the zeros of the _complex_ lifted system is performed by reducing the corresponding system pencil 
to an appropriate Kronecker-like form which exhibits the finite and infinite eigenvalues. 
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For a system `psys` of period `T`, 
the finite zeros are determined as those eigenvalues which have imaginary parts in the interval `[-ω/2, ω/2]`, where `ω = 2π/(P*T)`.
To eliminate possible spurious finite eigenvalues, the intersection of two finite eigenvalue sets is computed 
for two lifted systems obtained for `N` and `N+2` harmonic components.    
The infinite zeros are determined as the infinite zeros of the LTI system `(A(ti), B(ti), C(ti), D(ti))` 
resulting for a random time value `ti`. _Warning:_ While this evaluation of the number of infinite zeros mostly 
provides the correct result, there is no theoretical assessment of this approach (counterexamples are welcome!). 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances for the nonzero
elements of the underlying lifted system pencil, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of the pencil, and `ϵ` is the 
working machine epsilon. 
"""
function pszero(psys::PeriodicStateSpace{<: HarmonicArray}, N::Union{Int,Missing} = missing; P::Int= 1, fast::Bool = true, atol::Real = 0, rtol::Real = 0) 
    ismissing(N) && (N = max(20, max(size(psys.A.values,3),size(psys.B.values,3),size(psys.C.values,3),size(psys.D.values,3))-1))
    (N == 0 || islti(psys) ) && (return MatrixPencils.spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])
    period = psys.A.period
    ωhp2 = pi/P/period
    n = size(psys.A,1)
    T = promote_type(Float64, eltype(psys.A))
    # employ heuristics to determine fix finite zeros by comparing two sets of computed zeros
    z = MatrixPencils.spzeros(dssdata(ps2fls(psys, N; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
    zf = z[isfinite.(z)]
    ind = sortperm(imag(zf),by=abs); 
    nf = count(abs.(imag(zf[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf = zf[ind[1:nf]]
    z2 = MatrixPencils.spzeros(dssdata(ps2fls(psys, N+2; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
    zf2 = z2[isfinite.(z2)]
    ind = sortperm(imag(zf2),by=abs); 
    nf2 = count(abs.(imag(zf2[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf2 = zf2[ind[1:nf2]]
    σf = Complex{T}[]
    nf < nf2 || ((zf, zf2) = (zf2, zf))
    atol > 0 || (norms = max(norm(psys.A.values,Inf),norm(psys.B.values,Inf),norm(psys.C.values,Inf),norm(psys.D.values,Inf)))
    tol = atol > 0 ? atol : (rtol > 0 ? rtol*norms : sqrt(eps(T))*norms)
    for i = 1:min(nf,nf2)
        minimum(abs.(zf2 .- zf[i])) < tol  && push!(σf,zf[i])
    end
    isreal(σf) && (σf = real(σf))
    if any(isinf.(z)) 
       # Conjecture: The number of infinite zeros is the same as that of the time-evaluated system! 
       zm = MatrixPencils.spzeros(dssdata(psteval(psys, period*rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
       return [σf; zm[isinf.(zm)]]
    else
       return σf
    end
end
"""
     pszero(psys::PeriodicStateSpace{PeriodicMatrix}[, K]; atol, rtol, fast) -> val
     pszero(psys::PeriodicStateSpace{PeriodicArray}[, K]; atol, rtol, fast) -> val

Compute the finite and infinite zeros of a discrete-time periodic system `psys = (A(t), B(t), C(t), D(t))` in `val`, 
where the periodic system matrices `A(t)`, `B(t)`, `C(t)`, and `D(t)` are in either _Periodic Matrix_ or _Periodic Array_ representation. 
The optional argument `K` specifies a desired time to start the sequence of periodic matrices (default: `K = 1`).

The computation of zeros relies on the _fast_ structure exploiting reduction of singular periodic pencils as described in [1], 
which separates a matrix pencil `M-λN` which contains the infinite and finite zeros 
and the left and right Kronecker indices.
The reduction is performed using orthonal similarity transformations and involves rank decisions based 
on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. 

The keyword arguments `atol` and `rtol` specify the absolute and relative tolerances for the nonzero
elements of the periodic system matrices, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is proportional with the largest dimension of the periodic matrices, 
and `ϵ` is the working machine epsilon. 

_References_

[1] A. Varga and P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems & Control Letters 50:371–381, 2003.
"""
function pszero(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}, K::Int = 1; fast::Bool = true, atol::Real = 0, rtol::Real = 0) where {T}
    islti(psys)  && (return MatrixPencils.spzeros(psys.A.M[1], I, psys.B.M[1], psys.C.M[1], psys.D.M[1]; fast, atol1 = atol, atol2 = atol, rtol)[1])

    (na, nb, nc, nd) = (psys.A.dperiod, psys.B.dperiod, psys.C.dperiod, psys.D.dperiod)
    N = na*psys.A.nperiod
    p, m = size(psys)
    ndx, nx = size(psys.A)
    patype = length(nx) == 1 
    #si = [getpm(psys.A,K,na) getpm(psys.B,K,nb); getpm(psys.C,K,nc) getpm(psys.D,K,nd)]
    si = [psys.A[K] psys.B[K]; psys.C[K] psys.D[K]]
    ndxi = ndx[patype ? 1 : mod(K-1,na)+1]
    nxi1 = nx[patype ? 1 : mod(K,na)+1]
    ti = [ -I zeros(T,ndxi,m); zeros(T,p,nxi1+m)]
    tolr = atol
    atol == 0 && (sc = sum(nx.+p)* eps(float(T)))
    n1 = size(si,2)
    for i = K:K+N-3
        m1 = size(si,1)
        #si1 = [getpm(psys.A,i+1,na) getpm(psys.B,i+1,nb); getpm(psys.C,i+1,nc) getpm(psys.D,i+1,nd)]
        si1 = [psys.A[i+1] psys.B[i+1]; psys.C[i+1] psys.D[i+1]]
        mi1 = size(si1,1)
        ndxi1 = mi1-p
        nxi2 = nx[patype ? 1 : mod(i+1,na)+1]
        ti1 = [ -I zeros(T,ndxi1,m); zeros(T,p,nxi2+m)]

        F = qr([ ti; si1 ], ColumnNorm()) 
        nr = minimum(size(F.R))
        # compute rank of r 
        ss = abs.(diag(F.R[1:nr,1:nr]))
        atol == 0 && ( tolr = sc * maximum(ss))
        rankr = count(ss .> tolr)
        si = F.Q'*[si; zeros(T,mi1,n1)]; si=si[rankr+1:end,:]
        ti = F.Q'*[ zeros(T,m1,nxi2+m); ti1]; ti = ti[rankr+1:end,:]
    end
    #sn = [getpm(psys.A,K+N-1,na) getpm(psys.B,K+N-1,nb); getpm(psys.C,K+N-1,nc) getpm(psys.D,K+N-1,nd)]
    sn = [psys.A[K+N-1] psys.B[K+N-1]; psys.C[K+N-1] psys.D[K+N-1]]
    ndxi = ndx[patype ? 1 : mod(K+N-2,na)+1]
    nxi1 = nx[patype ? 1 : mod(K+N-1,na)+1]
    tn = [ I zeros(T,ndxi,m); zeros(T,p,nxi1+m)]
    a = [ zeros(T,size(tn)...) sn; si ti] 
    e = [ tn zeros(T,size(sn)...); zeros(T,size(si)...) zeros(T,size(ti)...)] 
    return complex(pzeros(a, e; fast, atol1 = atol, atol2 = atol, rtol)[1]).^(1/N)    
end
function pszero(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, PeriodicSymbolicMatrix, PeriodicTimeSeriesMatrix}}, K::Union{Int,Missing} = missing; kwargs...) 
    return pszero(convert(PeriodicStateSpace{HarmonicArray},psys), K; kwargs...)
end
"""
     isstable(psys[, K = 1]; smarg = 1, fast = false, offset = sqrt(ϵ), kwargs...) -> Bool

Return `true` if the continuous-time periodic system `psys` has only stable characteristic multipliers and `false` otherwise. 

To assess the stability, the absolute values of the characteristic multipliers (i.e., the eigenvalues of the monodromy matrix)
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The monodromy matrix is determined as a product `K` state transition matrices (default: `K = 1`) 
computed by integrating numerically a homogeneous linear ODE with periodic coefficients.
If `fast = false` (default) then the characteristic multipliers are computed using an approach
based on the periodic Schur decomposition [1], while if `fast = true` 
the structure exploiting reduction [2] of an appropriate lifted pencil is employed. 
This later option may occasionally lead to inaccurate results for large number of matrices. 

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.
"""
function isstable(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:c,T}}, K::Int = 1; smarg::Real = 1, 
                  fast::Bool = false, offset::Real = sqrt(eps(float(real(T)))), kwargs...) where T
    ev = pseig(monodromy(convert(PeriodicFunctionMatrix,psys.A), K; kwargs...); fast)
    return all(abs.(ev) .< smarg-abs(offset))
end
"""
    isstable(psys; smarg = 1, fast = false, offset = sqrt(ϵ)) -> Bool

Return `true` if the discrete-time periodic system `psys` has only stable characteristic multipliers and `false` otherwise. 
    
To assess the stability, the absolute values of the characteristic multipliers (i.e., the eigenvalues of the monodromy matrix)
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `fast = false` (default) then the characteristic multipliers are computed using an approach
based on the periodic Schur decomposition [1], while if `fast = true` 
the structure exploiting reduction [2] of an appropriate lifted pencil is employed. 
This later option may occasionally lead to inaccurate results for large number of matrices. 

_References_

[1] A. Bojanczyk, G. Golub, and P. Van Dooren, 
    The periodic Schur decomposition. Algorithms and applications, Proc. SPIE 1996.

[2] A. Varga & P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems and Control Letters, 50:371-381, 2003.
"""
function isstable(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}; smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps(float(real(T))))) where T
    ev = pseig(psys.A; fast)
    return all(abs.(ev) .< smarg-abs(offset))
end
"""
    psh2norm(psys; adj = false, smarg = 1, fast = false, offset = sqrt(ϵ)) -> nrm

Compute the H2-norm of a discrete-time periodic system `psys = (A(t),B(t),C(t),D(t))`.  
For the computation of the norm, the formulas given in [1] are employed. 
For `adj = false` the norm is computed as
 
     nrm = sqrt(sum(tr(C(t)P(t)C(t)'+D(t)*D(t)'))),

where `P(t)` is the controllability Gramian satisfying the periodic Lyapunov equation

     P(t+1) = A(t)P(t)A(t)' + B(t)B(t)',

while for `adj = true` the norm is computed as
 
    nrm = sqrt(sum(tr(B(t)'Q(t+1)B(t)+D(t)'*D(t)))),

where `Q(t)` is the observability Gramian satisfying the periodic Lyapunov equation

    Q(t) = A(t)'Q(t+1)A(t) + C(t)'C(t) .

The norm is set to infinity for an unstable system.
    
To assess the stability, the absolute values of the characteristic multipliers of `A(t)` 
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `fast = false` (default) then the norm is evaluated using an approach based on the periodic Schur decomposition of `A(t)`, 
while if `fast = true` the norm of the lifted standard system is evaluated.  
This later option may occasionally lead to inaccurate results for large number of matrices. 

_References_

[1] S. Bittanti and P. Colaneri. Periodic Systems : Filtering and Control.
    Springer-Verlag London, 2009. 
"""
function psh2norm(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps(float(real(T))))) where {T}
    !isstable(psys; smarg, offset, fast ) && (return Inf)  # unstable system
    if fast
       return gh2norm(ps2ls(psys, ss = true))
    else
       if adj 
        W = psys.C'*psys.C
        Q = pdlyap(psys.A, (W+W')/2,adj=true)
          return sqrt(trace(psys.B'*pmshift(Q,1)*psys.B+psys.D'*psys.D))
          #return sqrt(sum(tr(psys.B'*pmshift(Q,1)*psys.B+psys.D'*psys.D)))
       else
          W = psys.B*psys.B'
          P = pdlyap(psys.A, (W+W')/2,adj=false)
          return sqrt(trace(psys.C*P*psys.C'+psys.D*psys.D'))
          #return sqrt(sum(tr(psys.C*P*psys.C'+psys.D*psys.D')))
       end
    end
end
"""
    psh2norm(psys, K; adj = false, smarg = 1, fast = false, offset = sqrt(ϵ), solver = "auto", reltol = 1.e-4, abstol = 1.e-7, quad = false) -> nrm

Compute the H2-norm of a continuous-time periodic system `psys = (A(t),B(t),C(t),D(t))`.  
For the computation of the norm, the formulas given in [1] are employed, 
in conjunction with the multiple-shooting approach of [2] using `K` discretization points.  
For a periodic system of period  `T`, for `adj = false` the norm is computed as
 
     nrm = sqrt(Integral(tr(C(t)P(t)C(t)')))dt/T),

where `P(t)` is the controllability Gramian satisfying the periodic differential Lyapunov equation

     .
     P(t) = A(t)P(t)A(t)' + B(t)B(t)',

while for `adj = true` the norm is computed as
 
    nrm = sqrt(Integral(tr(B(t)'Q(t)B(t)))dt/T),

where Q(t) is the observability Gramian satisfying the periodic differential Lyapunov equation

     .
    -Q(t) = A(t)'Q(t)A(t) + C(t)'C(t) .

If `quad = true`, a simple quadrature formula based on the sum of time values is employed (see [2]).
This option ensures a faster evaluation, but is potentially less accurate then the exact evaluation
employed if `quad = false` (default). 

The norm is set to infinity for an unstable system or for a non-zero `D(t)`.
    
To assess the stability, the absolute values of the characteristic multipliers of `A(t)` 
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 
If `fast = false` (default) then the stability is checked using an approach based on the periodic Schur decomposition of `A(t)`, 
while if `fast = true` the stability is checked using a lifting-based approach.  

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

_References_

[1] P. Colaneri. Continuous-time periodic systems in H2 and H∞: Part I: Theoretical Aspects.
    Kybernetika, 36:211-242, 2000. 

[2] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function psh2norm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, HarmonicArray}}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7, quad = false) 
    norm(psys.D) == 0 || (return Inf)            
    !isstable(psys, K; smarg, offset, fast, solver, reltol, abstol) && (return Inf)  # unstable system
    P = adj ? pgclyap(psys.A, psys.C'*psys.C, K; adj, solver, reltol, abstol) : pgclyap(psys.A, psys.B*psys.B', K; adj, solver, reltol, abstol)
    Ts = psys.period/K/P.nperiod
    pp = length(P)
    if quad 
       # use simple quadrature formula 
       nrm = 0
       if adj
          for i = K:-1:1
              ip =  mod.(i-1,pp).+1
              Bt = tpmeval(psys.B,(i-1)*Ts)
              nrm += tr(Bt'*P.values[ip]*Bt)
          end
       else
          for i = 1:K
              ip =  mod.(i-1,pp).+1
              Ct = tpmeval(psys.C,(i-1)*Ts)
              temp = tr(Ct*P.values[ip]*Ct')
              # (i == 1 || i == K) && (temp = temp/2) # for some reason the trapezoidal method is less accurate 
              nrm += temp
          end
       end
       return sqrt(nrm*Ts*P.nperiod/psys.period)
    end
    μ = Vector{eltype(psys)}(undef,K)
    if adj
       #Threads.@threads for i = K:-1:1
        for i = K:-1:1
            ip =  mod.(i-1,pp).+1
            iw = ip < pp ? ip+1 : 1 
            @inbounds μ[i] = tvh2norm(psys.A, psys.B, psys.C, P.values[iw], (i-1)*Ts, i*Ts; adj, solver, reltol, abstol)
        end
    else
       #Threads.@threads for i = K:-1:1
       for i = 1:K
           ip =  mod.(i-1,pp).+1
           @inbounds μ[i]  = tvh2norm(psys.A, psys.B, psys.C, P.values[ip], i*Ts, (i-1)*Ts; adj, solver, reltol, abstol) 
       end
    end
    return sqrt(sum(μ)*P.nperiod/psys.period)
end
function psh2norm(psys::PeriodicStateSpace{<:PeriodicSymbolicMatrix}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7, quad = false) 
    psh2norm(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys), K; adj, smarg, fast, offset, solver, reltol, abstol, quad) 
end
function psh2norm(psys::PeriodicStateSpace{<:PeriodicTimeSeriesMatrix}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7, quad = false) 
    psh2norm(convert(PeriodicStateSpace{HarmonicArray},psys), K; adj, smarg, fast, offset, solver, reltol, abstol, quad) 
end

function tvh2norm(A::PM1, B::PM2, C::PM3, P::AbstractMatrix, tf, t0; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
    {PM1 <: Union{PeriodicFunctionMatrix,HarmonicArray,PeriodicTimeSeriesMatrix}, 
     PM2 <: Union{PeriodicFunctionMatrix,HarmonicArray,PeriodicTimeSeriesMatrix}, 
     PM3 <: Union{PeriodicFunctionMatrix,HarmonicArray,PeriodicTimeSeriesMatrix}} 
    """
       tvh2norm(A, B, C, P, tf, to; adj, solver, reltol, abstol) ->  μ 
 
    Cmputes the H2-norm of the system (A(t),B(t),C(t),0) by integrating for tf > t0 and adj = false
    jointly the differential matrix Lyapunov equation

            . 
            W(t) = A(t)*W(t)+W(t)*A'(t)+B(t)B(t)', W(t0) = P

    and

            .
            h(t) = trace(C(t)W(t)C'(t)),   h(t0) = 0;
                     
 
    or for tf < t0 and adj = true

            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t)'C(t), W(t0) = P

    and

            .
            h(t) = -trace(B(t)'W(t)B(t)),   h(t0) = 0.


    The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
    together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
    absolute accuracy `abstol` (default: `abstol = 1.e-7`).
    """
    n = size(A,1)
    n == size(A,2) || error("the periodic matrix A must be square")
    n == size(C,2) || error("the periodic matrix C must have same number of columns as A")
    T = promote_type(typeof(t0), typeof(tf))
    # using OrdinaryDiffEq
    u0 = [MatrixEquations.triu2vec(P);zero(T)]
    tspan = (T(t0),T(tf))
    fclyap1!(du,u,p,t) = adj ? muladdcsym1!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)', tpmeval(B,t)') : 
                               muladdcsym1!(du, u, 1, tpmeval(A,t), tpmeval(B,t), tpmeval(C,t)) 
    prob = ODEProblem(fclyap1!, u0, tspan)
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
    else 
       if reltol > 1.e-4  
          # low accuracy automatic selection
          sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
       else
          # high accuracy automatic selection
          sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
       end
    end
    return sol(tf)[end] 
end
function muladdcsym1!(y::AbstractVector, x::AbstractVector, isig, A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    # isig*(A*X + X*A' + B*B')
    # isig*tr(C*X*C')
    n, m = size(B)
    T1 = promote_type(eltype(A), eltype(x))
    # TO DO: eliminate building of X by using directly x
    X = MatrixEquations.vec2triu(convert(AbstractVector{T1}, view(x,1:length(x)-1)), her=true)
    @inbounds begin
       k = 1
       for j = 1:n
          for i = 1:j
             temp = zero(T1)
             #temp = B[i,j]
             for l = 1:m
                 temp += B[i,l] * B[j,l]
             end
             for l = 1:n
                temp += A[i,l] * X[l,j] + X[i,l] * A[j,l]
             end
             y[k] = isig*temp
             k += 1
          end
       end
    end
   #  @inbounds begin
   #      k = 1
   #      for j = 1:n
   #         for i = 1:j
   #            temp = 0
   #            for l = 1:n
   #               temp += X[l,i] * C[l,j]
   #            end
   #            y[k] = temp
   #            k += 1
   #         end
   #      end
   #   end
   y[k] = isig*tr(C*X*C') 
   return y
end
"""
    pshanorm(psys; smarg = 1, lifting = false, offset = sqrt(ϵ)) -> nrm

Compute the Hankel-norm of a stable discrete-time periodic system `psys = (A(t),B(t),C(t),D(t))`.  
For the computation of the norm, the formulas given in [1] are employed. 
The Hankel norm is computed as
 
     nrm = maximum(sqrt(Λ(P(t)Q(t))),

where `P(t)` is the controllability Gramian satisfying the periodic Lyapunov equation

     P(t+1) = A(t)P(t)A(t)' + B(t)B(t)',

and `Q(t)` is the observability Gramian satisfying the periodic Lyapunov equation

    Q(t) = A(t)'Q(t+1)A(t) + C(t)'C(t) .
   
To assess the stability, the absolute values of the characteristic multipliers of `A(t)` 
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

If `lifting = false` (default), then the norm is evaluated using an approach based on the periodic Schur decomposition of `A(t)`, 
while if `lifting = true` the norm of the lifted cyclic system is evaluated.  
This later option may lead to excessive computational times for large matrices or large periods. 

_References_

[1] S. Bittanti and P. Colaneri. Periodic Systems : Filtering and Control.
    Springer-Verlag London, 2009. 
"""
function pshanorm(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}; smarg::Real = 1, lifting::Bool = false, 
                offset::Real = sqrt(eps(float(real(T))))) where {T}
  !isstable(psys; smarg, offset, fast = lifting ) && error("The system must be stable")  # unstable system
  if lifting
     return ghanorm(ps2ls(psys, cyclic = true))[1]
  else
     Y = psys.C'*psys.C
     Q = pdlyap(psys.A,(Y+Y')/2,adj=true)
     Y = psys.B*psys.B'
     P = pdlyap(psys.A,(Y+Y')/2,adj=false)
     Y = P*Q
     l = length(Y)
     if typeof(psys.A) <: PeriodicArray
        return maximum(sqrt.([maximum(abs.(eigvals(Y.M[:,:,i]))) for i in 1:l]))
     else
        return maximum(sqrt.([maximum(abs.(eigvals(Y.M[i]))) for i in 1:l]))
     end
  end
end
"""
    pshanorm(psys, K; smarg = 1, offset = sqrt(ϵ), solver = "auto", reltol = 1.e-4, abstol = 1.e-7) -> nrm

Compute the Hankel-norm of a stable continuous-time periodic system `psys = (A(t),B(t),C(t),D(t))`.  
For the computation of the norm, the approach suggested in [1] is employed, 
in conjunction with the multiple-shooting approach using `K` discretization points.  
For a periodic system of period  `T`, the Hankel-norm is defined as
 
     nrm = sqrt(max(Λ(P(t)Q(t)))), for t ∈ [0,T]

where `P(t)` is the controllability Gramian satisfying the periodic differential Lyapunov equation

     .
     P(t) = A(t)P(t)A(t)' + B(t)B(t)',

and `Q(t)` is the observability Gramian satisfying the periodic differential Lyapunov equation

     .
    -Q(t) = A(t)'Q(t)A(t) + C(t)'C(t) .

The norm is evaluated from the `K` time values of `P(t)` and `Q(t)` in the interval `[0,T]` and 
the precision is (usually) better for larger values of `K`.
   
To assess the stability, the absolute values of the characteristic multipliers of `A(t)` 
must be less than `smarg-β`, where `smarg` is the discrete-time stability margin (default: `smarg = 1`)  and 
`β` is an offset specified via the keyword parameter `offset = β` to be used to numerically assess the stability
of eigenvalues. The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 


_References_

[1] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function pshanorm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, HarmonicArray, PeriodicSymbolicMatrix}}, K::Int = 1; smarg::Real = 1, 
                  offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7) 
    !isstable(psys, K; smarg, offset, solver, reltol, abstol) && error("The system must be stable")  # unstable system
    Q = pgclyap(psys.A, psys.C'*psys.C, K; adj = true, solver, reltol, abstol) 
    P = pgclyap(psys.A, psys.B*psys.B', K; adj = false, solver, reltol, abstol)
    return sqrt(maximum(norm.(eigvals(P*Q),Inf)))
end
# function pshanorm(psys::PeriodicStateSpace{<:PeriodicSymbolicMatrix}, K::Int = 1; smarg::Real = 1, 
#                   offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7) 
#     pshanorm(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys), K; smarg, offset, solver, reltol, abstol, dt) 
# end
function pshanorm(psys::PeriodicStateSpace{<:PeriodicTimeSeriesMatrix}, K::Int = 1; smarg::Real = 1, 
                  offset::Real = sqrt(eps()), solver = "auto", reltol = 1e-4, abstol = 1e-7) 
    pshanorm(convert(PeriodicStateSpace{HarmonicArray},psys), K; smarg, offset, solver, reltol, abstol) 
end
"""
    pslinfnorm(psys, hinfnorm = false, rtolinf = 0.001, fast = true, offset = sqrt(ϵ)) -> (linfnorm, fpeak)
    pshinfnorm(psys, rtolinf = 0.001, fast = true, offset = sqrt(ϵ)) -> (hinfnorm, fpeak)

Compute for a discrete-time periodic system `psys = (A(t),B(t),C(t),D(t)` with the lifted transfer function  matrix `G(λ)` 
the `L∞` norm `linfnorm` with ` pslinfnorm` or the `H∞` norm `hinfnorm` with ` pshinfnorm` (i.e.,  the peak gain of `G(λ)`) and 
the corresponding peak frequency `fpeak`, where the peak gain is achieved.
If `hinfnorm = true`, the `H∞` norm is computed with ` pslinfnorm`.

The `L∞` norm is infinite if `psys` has poles on the stability domain boundary, 
i.e., on the unit circle. The `H∞` norm is infinite if `psys` has unstable poles. 

To check the lack of poles on the stability domain boundary, the poles of `psys` 
must not have moduli within the interval `[1-β,1+β]`, where `β` is the stability domain boundary offset.  
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

The keyword argument `rtolinf` specifies the relative accuracy for the computed infinity norm. 
The  default value used for `rtolinf` is `0.001`.

The computation of the `L∞` norm is based on the algorithm proposed in [1]. 
The involved computations of characteristic multipliers are performed either with the fast reduction method of [2], 
if `fast = true` or if time-varying dimensions are present, 
or the generalized periodic Schur decomposition based method of [3], if `fast = false`.  

_References_

[1] A. Varga. "Computation of L∞-norm of linear discrete-time periodic systems." Proc. MTNS, Kyoto, 2007.

[2] A. Varga and P. Van Dooren. Computing the zeros of periodic descriptor systems.
    Systems & Control Letters 50:371–381, 2003.

[3] Kressner, D.
    An efficient and reliable implementation of the periodic QZ
    algorithm. IFAC Workshop on Periodic Control Systems (PSYCO
    2001), Como (Italy), August 27-28 2001. Periodic Control
    Systems 2001 (IFAC Proceedings Volumes), Pergamon.
"""   
function pslinfnorm(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}; hinfnorm::Bool = false, rtolinf::Real = float(real(T))(0.001), fast::Bool = true, 
                   offset::Real = sqrt(eps(float(real(T)))))  where {T} 
    
    T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
    ZERO = real(T1)(0)

    # detect zero case
    # iszero(sys, atol1 = atol1, atol2 = atol2, rtol = rtol) && (return ZERO, ZERO)

    # quick exit for zero dimensions  
    ny, nu = size(psys)
    (nu == 0 || ny == 0) && (return ZERO, ZERO)

    # quick exit in constant case  
    size(psys.A,1) == 0 && (return opnorm(ps2ls(psys).D), ZERO)

    β = abs(offset)
    Ts = abs(psys.Ts)
    
    # check for poles on the boundary of the stability domain
    ft = psceig(psys.A) 
    hinfnorm && any(abs.(ft) .> 1-β) && (return Inf, NaN)
    for i = 1:length(ft)
        abs(ft[i]) >= 1-β && abs(ft[i]) <= 1+β && (return Inf, abs(log(complex(ft[i]))/Ts))
    end
    
    # compute L∞-norm according to system type
    return psnorminfd(psys, ft, Ts, rtolinf, fast)
    # end PSLINFNORM
end
function psnorminfd(psys::PeriodicStateSpace{<: PeriodicMatrix}, ft0, Ts, tol, fast) 

    T = eltype(psys)
    ny, nu = size(psys)
    min(ny, nu) == 0 && (return T(0), T(0))
 
    # Discrete-time L∞ norm computation
    # It is assumed that psys has no poles on the unit circle
 
    # Tolerance for detection of unit circle modes
    epsm = eps(T)
    toluc1 = 100 * epsm       # for simple roots
    toluc2 = 10 * sqrt(epsm)  # for double root
    
    # Problem dimensions
    ndx, nx = size(psys.A)
    ny, nu = size(psys)
    
    # Build a new vector TESTFRQ of test frequencies containing the peaking
    # frequency for each mode (or an approximation thereof for non-resonant modes).
    sr = log.(complex(ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)]));   # equivalent jw-axis modes
    #sr = ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)];   # equivalent jw-axis modes
    # asr2 = abs.(real(sr))   # magnitude of real parts
    w0 = abs.(sr);           # fundamental frequencies
 
    # ikeep = (imag.(sr) .>= 0) .& ( w0 .> 0)
    # testfrq = w0[ikeep].*sqrt.(max.(0.25,1 .- 2 .*(asr2[ikeep]./w0[ikeep]).^2))
    testfrq = [[0]; w0]
    
    # Back to unit circle, and add z = exp(0) and z = exp(pi)
    testz = [exp.(im*testfrq); [-1] ]
   
    gmin = 0
    fpeak = 0
    sys = ps2spls(psys)    
    T1 = promote_type(Float64, eltype(testz), T)        
 
    # Compute lower estimate GMIN as max. gain over the selected frequencies
    for i = 1:length(testz)
        z = testz[i]
        #gw = opnorm(pseval(psys,z))
        gw = opnorm(sys.C*(lu!(T1(z)*sys.E-sys.A)\Matrix(sys.B))+sys.D)
        gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
    end
    gmin == 0 && (return T(0), T(0))
    (pa, pb, pc, pd) = (psys.A.dperiod, psys.B.dperiod, psys.C.dperiod, psys.D.dperiod)
    p = lcm(pa,pb,pc,pd)
    nx2 = nx+nx
    H = Vector{Matrix{Float64}}(undef,p)
    J = Vector{Matrix{Float64}}(undef,p)
 
    # Modified gamma iterations (Bruinsma-Steinbuch algorithm) starts:
    iter = 1;
    while iter < 30
       # Test if G = (1+TOL)*GMIN qualifies as upper bound
       g = (1+tol) * gmin;
       # Compute the finite eigenvalues of the symplectic pencil
       # deflate nu+ny simple infinite eigenvalues
       k = argmin(nx); #k = 1
       iam1 = mod(pa+k-2,pa)+1
       ii = 1
       for i = k:p+k-1
           ia = mod(i-1,pa)+1
           #iam1 = mod(i-2,pa)+1
           iap1 = mod(i,pa)+1
           ib = mod(i-1,pb)+1
           ic = mod(i-1,pc)+1
           id = mod(i-1,pd)+1
           ni = nx[ia]; nip1 = nx[iap1]
           nui1 = ndx[ia]; nuim1 = ndx[iam1]
           #nx2i = ni+nuim1
           h1 = [psys.A.M[ia] zeros(T,nui1,ni); 
                 zeros(T,ni,ni) I;
                 zeros(nu,ni+ni); 
                 psys.C.M[ic] zeros(T,ny,ni)]
           h2 = [psys.B.M[ib] zeros(T,nui1,ny);
                 zeros(T,ni,nu) psys.C.M[ic]';
                 I psys.D.M[id]'; 
                 psys.D.M[id] g^2*I]
           j1 =  [I zeros(T,nip1,nip1); 
                  zeros(T,ni,nip1) psys.A.M[ia]';
                  zeros(T,nu,nip1) psys.B.M[ib]';
                  zeros(T,ny,nip1+nip1) ]
           _, tau = LinearAlgebra.LAPACK.geqrf!(h2)
           LinearAlgebra.LAPACK.ormqr!('L','T',h2,tau,h1)
           LinearAlgebra.LAPACK.ormqr!('L','T',h2,tau,j1)
           i1 = nu+ny+1:nu+ny+nui1+ni
           H[ii] = view(h1,i1,:)
           J[ii] = view(j1,i1,:)
           iam1 = ia 
           ii = ii+1
       end
       if !fast && nx[k] !== maximum(nx)
          @warn "only constant dimensions are supported: fast approach employed"
          fast = true
       end
       heigs = fast ? eigvals!(psreduc_reg(H,J)...) : PeriodicMatrices.pschur(H, J, withZ = false)[5]
       heigs =  heigs[abs.(heigs) .< 1/toluc2]
 
       # Detect unit-circle eigenvalues
       mag = abs.(heigs)
       uceig = heigs[abs.(1 .- mag) .< toluc2 .+ toluc1*mag]
    
       # Compute frequencies where gain G is attained and
       # generate new test frequencies
       ang = sort(angle.(uceig));
       ang = unique(max.(epsm,ang[ang .> 0]))
       lan0 = length(ang);
       if lan0 == 0
          # No unit-circle eigenvalues for G = GMIN*(1+TOL): we're done
          return gmin, fpeak/Ts
        else
          lan0 == 1 && (ang = [ang;ang])   # correct pairing
          lan = length(ang)
       end
    
       # Form the vector of mid-points and compute
       # gain at new test frequencies
       gmin0 = gmin;   # save current lower bound
       #testz = exp.(im*((ang[1:lan-1]+ang[2:lan])/2))
       # Compute lower estimate GMIN as max. gain over the selected frequencies
       
       for i = 1:lan-1
           z = exp(im*((ang[i]+ang[i+1])/2))
           #gw = opnorm(pseval(psys,z))
           gw = opnorm(sys.C*(lu!(T1(z)*sys.E-sys.A)\Matrix(sys.B))+sys.D)
           gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
       end
     
       # If lower bound has not improved, exit (safeguard against undetected
       # unit-circle eigenvalues).
       (lan0 < 2 || gmin < gmin0*(1+tol/10)) && (return gmin, fpeak/Ts)
       iter += 1
    end
end  
function psnorminfd(psys::PeriodicStateSpace{<: PeriodicArray}, ft0, Ts, tol, fast) 

   T = eltype(psys)
   ny, nu = size(psys)
   min(ny, nu) == 0 && (return T(0), T(0))

   # Discrete-time L∞ norm computation
   # It is assumed that psys has no poles on the unit circle

   # Tolerance for detection of unit circle modes
   epsm = eps(T)
   toluc1 = 100 * epsm       # for simple roots
   toluc2 = 10 * sqrt(epsm)  # for double root
   
   # Problem dimensions
   nx = size(psys.A,1);
   ny, nu = size(psys.D)
   
   # Build a new vector TESTFRQ of test frequencies containing the peaking
   # frequency for each mode (or an approximation thereof for non-resonant modes).
   sr = log.(complex(ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)]));   # equivalent jw-axis modes
   #sr = ft0[(ft0 .!= 0) .& (abs.(ft0) .<= pi/Ts)];   # equivalent jw-axis modes
   # asr2 = abs.(real(sr))   # magnitude of real parts
   w0 = abs.(sr);           # fundamental frequencies

   # ikeep = (imag.(sr) .>= 0) .& ( w0 .> 0)
   # testfrq = w0[ikeep].*sqrt.(max.(0.25,1 .- 2 .*(asr2[ikeep]./w0[ikeep]).^2))
   testfrq = [[0]; w0]
   
   # Back to unit circle, and add z = exp(0) and z = exp(pi)
   testz = [exp.(im*testfrq); [-1] ]
  
   gmin = 0
   fpeak = 0
   sys = ps2spls(psys)    
   T1 = promote_type(Float64, eltype(testz), T)        

   # Compute lower estimate GMIN as max. gain over the selected frequencies
   for i = 1:length(testz)
       z = testz[i]
       #gw = opnorm(pseval(psys,z))
       gw = opnorm(sys.C*(lu!(T1(z)*sys.E-sys.A)\Matrix(sys.B))+sys.D)
       gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
   end
   gmin == 0 && (return T(0), T(0))
   pa = size(psys.A.M,3)
   pb = size(psys.B.M,3)
   pc = size(psys.C.M,3)
   pd = size(psys.D.M,3)
   p = lcm(pa,pb,pc,pd)
   nx2 = nx+nx
   H = zeros(T,nx2,nx2,p)
   J = zeros(T,nx2,nx2,p)
   i1 = nu+ny+1:nu+ny+nx2
   i2 = 1:nx2

   # Modified gamma iterations (Bruinsma-Steinbuch algorithm) starts:
   iter = 1;
   while iter < 30
      # Test if G = (1+TOL)*GMIN qualifies as upper bound
      g = (1+tol) * gmin;
      # Compute the finite eigenvalues of the symplectic pencil
      # deflate nu+ny simple infinite eigenvalues
      for i = 1:p
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          ic = mod(i-1,pc)+1
          id = mod(i-1,pd)+1
          h1 = [view(psys.A.M,:,:,ia) zeros(T,nx,nx); 
                zeros(T,nx,nx) I;
                zeros(nu,nx2); 
                view(psys.C.M,:,:,ic) zeros(T,ny,nx)]
          h2 = [view(psys.B.M,:,:,ib) zeros(T,nx,ny);
                zeros(T,nx,nu) view(psys.C.M,:,:,ic)';
                I view(psys.D.M,:,:,id)'; 
                view(psys.D.M,:,:,id) g^2*I]
          j1 =  [I zeros(T,nx,nx); 
                 zeros(T,nx,nx) view(psys.A.M,:,:,ia)';
                 zeros(T,nu,nx) view(psys.B.M,:,:,ib)';
                 zeros(T,ny,nx2) ]
          _, tau = LinearAlgebra.LAPACK.geqrf!(h2)
          LinearAlgebra.LAPACK.ormqr!('L','T',h2,tau,h1)
          LinearAlgebra.LAPACK.ormqr!('L','T',h2,tau,j1)
          copyto!(view(H,:,:,i),view(h1,i1,i2))
          copyto!(view(J,:,:,i),view(j1,i1,i2))
      end
      heigs = fast ? eigvals!(psreduc_reg(H,J)...) : PeriodicMatrices.pschur(H, J, withZ = false)[5]
      heigs =  heigs[abs.(heigs) .< 1/toluc2]

      # Detect unit-circle eigenvalues
      mag = abs.(heigs)
      uceig = heigs[abs.(1 .- mag) .< toluc2 .+ toluc1*mag]
   
      # Compute frequencies where gain G is attained and
      # generate new test frequencies
      ang = sort(angle.(uceig));
      ang = unique(max.(epsm,ang[ang .> 0]))
      lan0 = length(ang);
      if lan0 == 0
         # No unit-circle eigenvalues for G = GMIN*(1+TOL): we're done
         return gmin, fpeak/Ts
       else
         lan0 == 1 && (ang = [ang;ang])   # correct pairing
         lan = length(ang)
      end
   
      # Form the vector of mid-points and compute
      # gain at new test frequencies
      gmin0 = gmin;   # save current lower bound
      #testz = exp.(im*((ang[1:lan-1]+ang[2:lan])/2))
      # Compute lower estimate GMIN as max. gain over the selected frequencies
      for i = 1:lan-1
          z = exp(im*((ang[i]+ang[i+1])/2))
          #gw = opnorm(pseval(psys,z))
          gw = opnorm(sys.C*(lu!(T1(z)*sys.E-sys.A)\Matrix(sys.B))+sys.D)
          gw > gmin && (gmin = gw;  fpeak = abs(angle(z)))
      end
    
      # If lower bound has not improved, exit (safeguard against undetected
      # unit-circle eigenvalues).
      (lan0 < 2 || gmin < gmin0*(1+tol/10)) && (return gmin, fpeak/Ts)
      iter += 1
   end
end  
function pshinfnorm(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}; 
    rtolinf::Real = float(real(T))(0.001), fast::Bool = true, offset::Real = sqrt(eps(float(real(T)))))  where {T} 
    return pslinfnorm(psys; hinfnorm = true, rtolinf, fast, offset)
end
"""
    pseval(psys, val) -> Gval

Evaluate for a finite `λ = val`, the value `Gval` of the transfer function matrix `G(λ)` of the 
lifted system of the discrete-time periodic system `psys`. 
`val` must not be a pole of `psys`.
"""
function pseval(psys::PeriodicStateSpace{<: AbstractPeriodicArray{:d,T}}, val::Number) where {T}
    sys = ps2spls(psys)    
    T1 = promote_type(Float64, typeof(val), T)        
    return sys.C*(lu!(T1(val)*sys.E-sys.A)\Matrix(sys.B))+sys.D
end
"""
    pslinfnorm(psys, K = 100; hinfnorm = false, rtolinf = 0.001, offset = sqrt(ϵ), reltol, abstol, dt) -> (linfnorm, fpeak)
    pshinfnorm(psys, K = 100; rtolinf = 0.001, offset = sqrt(ϵ), reltol, abstol, dt) -> (linfnorm, fpeak)

Compute for a continuous-time periodic system `psys = (A(t),B(t),C(t),D(t)` the `L∞` norm `linfnorm` with `pslinfnorm` or the 
`H∞` norm `hinfnorm` with ` pshinfnorm` as defined in [1]. 
If `hinfnorm = true`, the `H∞` norm is computed with ` pslinfnorm`.
The corresponding peak frequency `fpeak`, where the peak gain is achieved, is usually not determined, excepting in some limiting cases.   
The `L∞` norm is infinite if `psys` has poles on the imaginary axis. 

To check the lack of poles on the imaginary axis, the characteristic exponents of `A(t)` 
must not have real parts in the interval `[-β,β]`, where `β` is the stability domain boundary offset.  
The offset  `β` to be used can be specified via the keyword parameter `offset = β`. 
The default value used for `β` is `sqrt(ϵ)`, where `ϵ` is the working machine precision. 

A bisection based algorith, as described in [2], is employed to approximate the `L∞` norm, and the keyword argument `rtolinf` specifies the relative accuracy for the computed infinity norm. 
The  default value used for `rtolinf` is `0.001`.

If `hinfnorm = true`, the `H∞` norm is computed. 
In this case, the stability of the system is additionally checked and 
the `H∞` norm is infinite for an unstable system.
To check the stability, the characteristic exponents of `A(t)` must have real parts less than `-β`.

The ODE solver to be employed to compute the characteristic multipliers of the system Hamiltonian can be specified using the keyword argument `solver` (default: `solver = "symplectic"`) 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and 
stepsize `dt` (default: `dt = 0`). The value stepsize is relevant only if `solver = "symplectic", in which case
an adaptive stepsize strategy is used if `dt = 0` and a fixed stepsize is used if `dt > 0`.
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "linear"` - use a special solver for linear ODEs (`MagnusGL6()`) with fixed time step `dt`;

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 


_References:_    

[1] P. Colaneri. Continuous-time periodic systems in H2 and H∞: Part I: Theoretical Aspects.
    Kybernetika, 36:211-242, 2000. 

[2] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""   
function pslinfnorm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix{:c,T}, HarmonicArray{:c,T}, PeriodicSymbolicMatrix{:c,T}}}, K::Int=100; hinfnorm::Bool = false, rtolinf::Real = float(real(T))(0.001), fast::Bool = true, 
                   offset::Real = sqrt(eps(float(real(T)))), solver = "symplectic", reltol = 1e-6, abstol = 1e-7, dt = 0)  where {T} 
    
    islti(psys) && (return glinfnorm(psaverage(psys); hinfnorm, rtolinf))      

    T1 = T <: BlasFloat ? T : (T <: Num ? Float64 : promote_type(Float64,T))
    ZERO = real(T1)(0)

    # detect zero case
    # iszero(sys, atol1 = atol1, atol2 = atol2, rtol = rtol) && (return ZERO, ZERO)

    # quick exit for zero dimensions  
    ny, nu = size(psys)
    (nu == 0 || ny == 0) && (return ZERO, ZERO)

    # quick exit in constant case  
    if PeriodicMatrices.isconstant(psys.D)
       gd = opnorm(tpmeval(psys.D,0),2)
    else
       f = t-> -opnorm(tpmeval(psys.D,t),2)
       gd = optimize(f,0,period,Optim.Brent(),rel_tol = eps()).minimum
    end
      
    size(psys.A,1) == 0 && (return gd, ZERO)

    β = abs(offset)
    epsm = eps(T1)
    toluc1 = 100 * epsm       # for simple roots
    toluc2 = 10 * sqrt(epsm)  # for double root
    
    # check for poles on the boundary of the stability domain
    ft = psceig(psys.A,K) 
    stable = all(real.(ft) .< -β)
    hinfnorm && !stable && (return Inf, NaN)
    for i = 1:length(ft)
        real(ft[i]) >= -β && real(ft[i]) <= β && (return Inf, T1 <: Complex ? imag(ft[i]) : abs(imag(ft[i])))
    end
    
    zeroD = gd == 0

    zeroD  && (iszero(psys.B) || iszero(psys.C)) && (return ZERO, ZERO)

    if stable
       gh = pshanorm(psys)
       gl = max(gd,gh)
       gu = gd + 2*gh
    else
       gl = gd; gu = glinfnorm(ps2fls(psys, 10); rtolinf)[1]
    end
    solver == "symplectic" && K < 10 && (K = 10; @warn "number of sampling values reset to K = $K")
    iter = 1
    while checkham1(psys,gu,K,toluc1, toluc2, solver, reltol, abstol, dt) && iter < 10
          gu *= 2
          iter += 1
    end
    # use bisection to determine    
    g = (gl+gu)/2
    while gu-gl > gu*rtolinf
          checkham1(psys,g,K,toluc1, toluc2, solver, reltol, abstol, dt) ? gl = g : gu = g   
          g = (gl+gu)/2
    end
    return g, nothing
end
function checkham1(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix{:c,T}, HarmonicArray{:c,T}, PeriodicSymbolicMatrix{:c,T}}}, g::Real, K::Int,toluc1, toluc2, solver, reltol, abstol, dt)  where {T}
    if iszero(psys.D) 
       Ht = [[psys.A (psys.B*psys.B')/g^2]; [-psys.C'*psys.C -psys.A']]
    else
       Rti = inv(g^2*I-psys.D'*psys.D)
       At = psys.A+psys.B*Rti*psys.D'*psys.C
       Gt = psys.B*Rti*psys.B'
       Qt = -psys.C'*(I+psys.D*Rti*psys.D')*psys.C
       Ht = [[At Gt]; [Qt -At']]
    end
    heigs = pseig(Ht, K; solver, reltol, abstol, dt)
    heigs =  heigs[abs.(heigs) .< 1/toluc2]

    # detect unit-circle eigenvalues
    mag = abs.(heigs)
    uceig = heigs[abs.(1 .- mag) .< toluc2 .+ toluc1*mag]
    return length(uceig) > 0
end
checkham1(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix{:c,T}, HarmonicArray{:c,T}, PeriodicSymbolicMatrix{:c,T}}}, g::Real, K::Int)  where {T} = checkham1(psys,g,K,eps(), sqrt(eps()), "symplectic", 1.e-10,1.e-10,0)
function pshinfnorm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix{:c,T}, HarmonicArray{:c,T}, FourierFunctionMatrix{:c,T},PeriodicSymbolicMatrix{:c,T}}}; 
    rtolinf::Real = float(real(T))(0.001), offset::Real = sqrt(eps(float(real(T)))), solver = "symplectic", reltol = 1e-6, abstol = 1e-7, dt = 0)  where {T} 
    return pslinfnorm(psys; hinfnorm = true, rtolinf, solver, reltol, abstol, dt)
end
