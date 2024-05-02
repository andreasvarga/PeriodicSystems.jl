"""
    pspole(psys::PeriodicStateSpace{PM}[,K]; kwargs...) -> val

Return for the periodic system `psys = (A(t),B(t),C(t),D(t))` the complex vector `val` containing 
the characteristic exponents of the periodic matrix `A(t)` (also called the _poles_ of the system `psys`). 

Depending on the underlying periodic matrix type `PM`, the optional argument `K` and keyword arguments `kwargs` may have the following values:

- if `PM = PeriodicFunctionMatrix`, or `PM = PeriodicSymbolicMatrix`, or `PM = PeriodicTimeSeriesMatrix`, then `K` is the number of factors used to express the monodromy matrix of `A(t)` (default: `K = 1`)  and `kwargs` are the keyword arguments of  [`pseig(::PeriodicFunctionMatrix)`](@ref); 

- if `PM = HarmonicArray`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  [`psceig(::HarmonicArray)`](@ref); 

- if `PM = FourierFunctionMatrix`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  [`psceig(::FourierFunctionMatrix)`](@ref); 

- if `PM = PeriodicMatrix` or `PM = PeriodicArray`, then `K` is the starting sample time (default: `K = 1`)  and `kwargs` are the keyword arguments of  [`psceig(::PeriodicMatrix)`](@ref); 
"""
pspole(psys::PeriodicStateSpace{<: PeriodicArray}, N::Int = 1; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicMatrix}, N::Int = 1; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicFunctionMatrix}, N::Int = 10; kwargs...) = psceig(psys.A, N; kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicTimeSeriesMatrix}, N::Int = 10; method = "cubic", kwargs...) = psceig(psys.A, N; method, kwargs...)
pspole(psys::PeriodicStateSpace{<: PeriodicSwitchingMatrix}) = psceig(psys.A)
pspole(psys::PeriodicStateSpace{<: SwitchingPeriodicMatrix}) = psceig(convert(PeriodicMatrix,psys.A))
pspole(psys::PeriodicStateSpace{<: HarmonicArray}, N::Int = 10; kwargs...) = psceighr(psys.A, N; kwargs...)
#pspole(psys::PeriodicStateSpace{<: HarmonicArray}, N::Int = 10; kwargs...) = psceig(psys.A, N; kwargs...)  # fallback version
pspole(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, N::Int = 10; kwargs...)  = psceigfr(psys.A, N; kwargs...)
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
a frequency-lifted LTI representation based on truncated block Toeplitz matrices (see [`ps2fls`](@ref)). 

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
    (N == 0 || islti(psys) ) && (return spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])
    period = psys.A.period
    ωhp2 = pi/P/period
    n = size(psys.A,1)
    T = promote_type(Float64, eltype(psys.A))
    # employ heuristics to determine fix finite zeros by comparing two sets of computed zeros
    z = spzeros(dssdata(ps2fls(psys, N; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
    zf = z[isfinite.(z)]
    ind = sortperm(imag(zf),by=abs); 
    nf = count(abs.(imag(zf[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf = zf[ind[1:nf]]
    z2 = spzeros(dssdata(ps2fls(psys, N+2; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
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
       zm = spzeros(dssdata(psteval(psys, period*rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
       return [σf; zm[isinf.(zm)]]
    else
       return σf
    end
end
"""
    pszero(psys::PeriodicStateSpece{FourierFunctionMatrix}[, N]; P, atol, rtol, fast) -> val

Compute the finite and infinite zeros of a continuous-time periodic system `psys = (Af(t), Bf(t), Cf(t), Df(t))` in `val`, 
where the periodic system matrices `Af(t)`, `Bf(t)`, `Cf(t)`, and `Df(t)` are in a _Fourier Function Matrix_ representation. 
`N` is the number of selected harmonic components in the Fourier series of the system matrices (default: `N = max(20,nh-1)`, 
where `nh` is the maximum number of harmonics terms) and the keyword parameter `P` is the number of full periods 
to be considered (default: `P = 1`) to build 
a frequency-lifted LTI representation based on truncated block Toeplitz matrices (see [`ps2frls`](@ref)). 

The computation of the zeros of the _real_ lifted system is performed by reducing the corresponding system pencil 
to an appropriate Kronecker-like form which exhibits the finite and infinite eigenvalues. 
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For a system `psys` of period `T`, 
the finite zeros are determined as those eigenvalues which have imaginary parts in the interval `[-ω/2, ω/2]`, where `ω = 2π/(P*T)`. 
To eliminate possible spurious finite eigenvalues, the intersection of two finite eigenvalue sets is computed 
for two lifted systems obtained for `N` and `N+2` harmonic components.    
The infinite zeros are determined as the infinite zeros of the LTI system `(Af(ti), Bf(ti), Cf(ti), Df(ti))` 
resulting for a random time value `ti`. _Warning:_ While this evaluation of the number of infinite zeros mostly 
provides the correct result, there is no theoretical assessment of this approach (counterexamples are welcome!). 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances for the nonzero
elements of the underlying lifted system pencil, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of the pencil, and `ϵ` is the 
working machine epsilon. 
"""
function pszero(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, N::Union{Int,Missing} = missing; P::Int= 1, fast::Bool = true, atol::Real = 0, rtol::Real = 0) 
    ismissing(N) && (N = max(20, maximum(ncoefficients.(Matrix(psys.A.M))), maximum(ncoefficients.(Matrix(psys.B.M))),
                                   maximum(ncoefficients.(Matrix(psys.C.M))), maximum(ncoefficients.(Matrix(psys.A.M)))))
    (N == 0 || islti(psys) ) && (return spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])

    # employ heuristics to determine fix finite zeros by comparing two sets of computed zeros
    z = spzeros(dssdata(ps2frls(psys, N; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 

    period = psys.A.period
    ωhp2 = pi/P/period
    n = size(psys.A,1)
    T = promote_type(Float64, eltype(psys.A))
    zf = z[isfinite.(z)]
    ind = sortperm(imag(zf),by=abs); 
    nf = count(abs.(imag(zf[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf = zf[ind[1:nf]]

    z2 = spzeros(dssdata(ps2frls(psys, N+2; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
    zf2 = z2[isfinite.(z2)]
    ind = sortperm(imag(zf2),by=abs); 
    nf2 = count(abs.(imag(zf2[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf2 = zf2[ind[1:nf2]]
    σf = Complex{T}[]
    nf < nf2 || ((zf, zf2) = (zf2, zf))
    atol > 0 || (norms = max(norm(coefficients(psys.A.M),Inf),norm(coefficients(psys.B.M),Inf),norm(coefficients(psys.C.M),Inf),norm(coefficients(psys.D.M),Inf)))
    tol = atol > 0 ? atol : (rtol > 0 ? rtol*norms : sqrt(eps(T))*norms)
    for i = 1:min(nf,nf2)
        minimum(abs.(zf2 .- zf[i])) < tol  && push!(σf,zf[i])
    end
    isreal(σf) && (σf = real(σf))

    if any(isinf.(z)) 
       # Conjecture: The number of infinite zeros is the same as that of the time-evaluated system! 
       zm = spzeros(dssdata(psteval(psys, period*rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
       zf = [σf; zm[isinf.(zm)]]
    end
    nz = length(zf)
    nz > n && (@warn "$(nz-n) spurious finite zero(s) present")
    return zf
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
    islti(psys)  && (return spzeros(psys.A.M[1], I, psys.B.M[1], psys.C.M[1], psys.D.M[1]; fast, atol1 = atol, atol2 = atol, rtol)[1])

    (na, nb, nc, nd) = (psys.A.dperiod, psys.B.dperiod, psys.C.dperiod, psys.D.dperiod)
    N = na*psys.A.nperiod
    p, m = size(psys)
    ndx, nx = size(psys.A)
    patype = length(nx) == 1 
    si = [getpm(psys.A,K,na) getpm(psys.B,K,nb); getpm(psys.C,K,nc) getpm(psys.D,K,nd)]
    ndxi = ndx[patype ? 1 : mod(K-1,na)+1]
    nxi1 = nx[patype ? 1 : mod(K,na)+1]
    ti = [ -I zeros(T,ndxi,m); zeros(T,p,nxi1+m)]
    tolr = atol
    atol == 0 && (sc = sum(nx.+p)* eps(float(T)))
    n1 = size(si,2)
    for i = K:K+N-3
        m1 = size(si,1)
        si1 = [getpm(psys.A,i+1,na) getpm(psys.B,i+1,nb); getpm(psys.C,i+1,nc) getpm(psys.D,i+1,nd)]
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
    sn = [getpm(psys.A,K+N-1,na) getpm(psys.B,K+N-1,nb); getpm(psys.C,K+N-1,nc) getpm(psys.D,K+N-1,nd)]
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
computed by integrating numerically a homogeneous linear ODE with periodic coefficients 
(see function [`monodromy`](@ref) for options which can be specified via the keyword arguments `kwargs`).
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
    psh2norm(psys, K; adj = false, smarg = 1, fast = false, offset = sqrt(ϵ), solver = "", reltol = 1.e-4, abstol = 1.e-7, quad = false) -> nrm

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
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and 
stepsize `dt' (default: `dt = 0`). The value stepsize is relevant only if `solver = "symplectic", in which case
an adaptive stepsize strategy is used if `dt = 0` and a fixed stepsize is used if `dt > 0`. (see also [`tvstm`](@ref)). 


_References_

[1] P. Colaneri. Continuous-time periodic systems in H2 and H∞: Part I: Theoretical Aspects.
    Kybernetika, 36:211-242, 2000. 

[2] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function psh2norm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, HarmonicArray, FourierFunctionMatrix}}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0, quad = false) 
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
    solver == "symplectic" && dt == 0 && (dt = K >= 100 ? Ts : Ts*K/100)
    if adj
       #Threads.@threads for i = K:-1:1
        for i = K:-1:1
            ip =  mod.(i-1,pp).+1
            iw = ip < pp ? ip+1 : 1 
            @inbounds μ[i] = tvh2norm(psys.A, psys.B, psys.C, P.values[iw], (i-1)*Ts, i*Ts; adj, solver, reltol, abstol, dt)
        end
    else
       #Threads.@threads for i = K:-1:1
       for i = 1:K
           ip =  mod.(i-1,pp).+1
           @inbounds μ[i]  = tvh2norm(psys.A, psys.B, psys.C, P.values[ip], i*Ts, (i-1)*Ts; adj, solver, reltol, abstol, dt) 
       end
    end
    #@show μ
    return sqrt(sum(μ)*P.nperiod/psys.period)
end
function psh2norm(psys::PeriodicStateSpace{<:PeriodicSymbolicMatrix}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0, quad = false) 
    psh2norm(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys), K; adj, smarg, fast, offset, solver, reltol, abstol, dt, quad) 
end
function psh2norm(psys::PeriodicStateSpace{<:PeriodicTimeSeriesMatrix}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0, quad = false) 
    psh2norm(convert(PeriodicStateSpace{HarmonicArray},psys), K; adj, smarg, fast, offset, solver, reltol, abstol, dt, quad) 
end

function tvh2norm(A::PM1, B::PM2, C::PM3, P::AbstractMatrix, tf, t0; adj = false, solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where
    {PM1 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicTimeSeriesMatrix}, 
     PM2 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicTimeSeriesMatrix}, 
     PM3 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicTimeSeriesMatrix}} 
    """
       tvh2norm(A, B, C, P, tf, to; adj, solver, reltol, abstol, dt) ->  μ 
 
    Cmputes the H2-norm of the system (A(t),B(t),C(t),0) by integrating tf > t0 and adj = false
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
    together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
    absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt' (default: `dt = abs(tf-t0)/100`, only used if `solver = "symplectic"`) (see [`tvstm`](@ref)). 
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
    elseif solver == "symplectic" 
       # high accuracy symplectic
       if dt == 0 
          sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, save_everystep = false)
          #@show sol.retcode
          if sol.retcode == :Failure
             sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = false, dt = abs(tf-t0)/100)
          end
       else
            sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = false, dt)
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
    pshanorm(psys, K; smarg = 1, offset = sqrt(ϵ), solver = "", reltol = 1.e-4, abstol = 1.e-7) -> nrm

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
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and 
stepsize `dt' (default: `dt = 0`). The value stepsize is relevant only if `solver = "symplectic", in which case
an adaptive stepsize strategy is used if `dt = 0` and a fixed stepsize is used if `dt > 0`. (see also [`tvstm`](@ref)). 


_References_

[1] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function pshanorm(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, HarmonicArray, FourierFunctionMatrix,PeriodicSymbolicMatrix}}, K::Int = 1; smarg::Real = 1, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) 
    !isstable(psys, K; smarg, offset, solver, reltol, abstol) && error("The system must be stable")  # unstable system
    Q = pgclyap(psys.A, psys.C'*psys.C, K; adj = true, solver, reltol, abstol) 
    P = pgclyap(psys.A, psys.B*psys.B', K; adj = false, solver, reltol, abstol)
    return sqrt(maximum(norm.(eigvals(P*Q),Inf)))
end
# function pshanorm(psys::PeriodicStateSpace{<:PeriodicSymbolicMatrix}, K::Int = 1; smarg::Real = 1, 
#                   offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) 
#     pshanorm(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys), K; smarg, offset, solver, reltol, abstol, dt) 
# end
function pshanorm(psys::PeriodicStateSpace{<:PeriodicTimeSeriesMatrix}, K::Int = 1; smarg::Real = 1, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) 
    pshanorm(convert(PeriodicStateSpace{HarmonicArray},psys), K; smarg, offset, solver, reltol, abstol, dt) 
end
