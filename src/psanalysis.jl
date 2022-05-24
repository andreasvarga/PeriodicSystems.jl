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
function pspole(psys::PeriodicStateSpace, K::Int; kwargs...)
    return psceig(psys.A,K; kwargs...)
end
pspole(psys::PeriodicStateSpace; kwargs...) = psceig(psys.A; kwargs...)
"""
    pszero(psys::PeriodicStateSpace{PM}[,K]; kwargs...) -> val

Return for the periodic system `psys = (A(t),B(t),C(t),D(t))` the complex vector `val` containing 
the finite and infinite zeros. 

Depending on the underlying periodic matrix type `PM`, the optional argument `K` and keyword arguments `kwargs` may have the following values:

- if `PM = HarmonicArray`, or `PM = PeriodicFunctionMatrix`, or `PM = PeriodicSymbolicMatrix`, or `PM = PeriodicTimeSeriesMatrix`, then `K` is the number of harmonic components used to represent the Fourier series of system matrices (default: `K = max(10,nh-1)`, `nh` is the maximum number of harmonics terms of the system mtrices)  and `kwargs` are the keyword arguments of  [`pszero(psys::PeriodicStateSpace{HarmonicArray})`](@ref); 

- if `PM = FourierFunctionMatrix`, then `K` is the number of harmonic components used to represent the Fourier series of system matrices (default: `K = max(10,nh-1)`, `nh` is the maximum number of harmonics terms of system matrices`)  and `kwargs` are the keyword arguments of  [`pszero(psys::PeriodicStateSpace{FourierFunctionMatrix})`](@ref); 

- if `PM = PeriodicMatrix` or `PM = PeriodicArray`, then `K` is the starting sample time (default: `K = 1`)  and `kwargs` are the keyword arguments of  [`pszero(psys::PeriodicStateSpace{PeriodicMatrix})`](@ref); 
"""
function pszero(psys::PeriodicStateSpace{<: Union{PeriodicFunctionMatrix, PeriodicSymbolicMatrix, PeriodicTimeSeriesMatrix}}, K::Union{Int,Missing} = missing; kwargs...) 
    return pszero(convert(PeriodicStateSpace{HarmonicArray},psys), K; kwargs...)
end
"""
    pszero(psys::PeriodicStateSpace{HarmonicArray}[, N]; P, atol, rtol, fast) -> val

Compute the finite and infinite zeros of a continuous-time periodic system `psys = (Ahr(t), Bhr(t), Chr(t), Dhr(t))` in `val`, 
where the periodic system matrices `Ahr(t)`, `Bhr(t)`, `Chr(t)`, and `Dhr(t)` are in a _Harmonic Array_ representation. 
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
The infinite zeros are determined as the infinite zeros of the LTI system `(Ahr(ti), Bhr(ti), Chr(ti), Dhr(ti))` 
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

# function isstable(psys::PeriodicStateSpace, smarg::Real = SYS.Ts == 0 ? 0 : 1; 
#                   fast = false, atol::Real = 0, atol1::Real = atol, atol2::Real = atol, 
#                   rtol::Real = SYS.nx*eps(real(float(one(T))))*iszero(min(atol1,atol2)), 
#                   offset::Real = sqrt(eps(float(real(T))))) where T
#     disc = (SYS.Ts != 0)
#     β = abs(offset); 
#     if SYS.E == I
#        isschur(SYS.A) ? poles = ordeigvals(SYS.A) : poles = eigvals(SYS.A)
#     else
#        poles = gpole(SYS; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol)
#        (any(isinf.(poles)) || any(isnan.(poles)))  && (return false)
#     end
#     return disc ? all(abs.(poles) .< smarg-β) : all(real.(poles) .< smarg-β)
# end
