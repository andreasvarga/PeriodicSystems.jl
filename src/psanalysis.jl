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
pspole(psys::PeriodicStateSpace{<: HarmonicArray}, N::Int = 10; kwargs...) = psceighr(psys.A, N; kwargs...)
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
