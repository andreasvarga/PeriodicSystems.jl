"""
    val = pspole(psys::PeriodicStateSpace{PM}[,K]; kwargs...) 

Return for the periodic system `psys = (A(t),B(t),C(t),D(t))` the complex vector `val` containing 
the characteristic exponents of the periodic matrix `A(t)` (also called the _poles_ of the system `psys`). 

Depending on the underlying periodic matrix type `PM`, the optional argument `K` and keyword arguments `kwargs` may have the following values:

- if `PM = PeriodicFunctionMatrix`, or `PM = PeriodicSymbolicMatrix`, or `PM = PeriodicTimeSeriesMatrix`, then `K` is the number of factors used to express the monodromy matrix of `A(t)` (default: `K = 1`)  and `kwargs` are the keyword arguments of  [`pseig(::PeriodicFunctionMatrix)`](@ref); 

- if `PM = HarmonicArray`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  [`psceig(::HarmonicArray)`](@ref); 

- if `PM = FourierFunctionMatrix`, then `K` is the number of harmonic components used to represent the Fourier series of `A(t)` (default: `K = max(10,nh-1)`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  [`psceig(::FourierFunctionMatrix)`](@ref); 

- if `PM = PeriodicMatrix` or `PM = PeriodicArray`, then `K` is the starting sample time (default: `K = 1`, `nh` is the number of harmonics terms of `A(t)`)  and `kwargs` are the keyword arguments of  [`psceig(::PeriodicMatrix)`](@ref); 
"""
function pspole(psys::PeriodicStateSpace, K::Int; kwargs...)
    return psceig(psys.A,K; kwargs...)
end
pspole(psys::PeriodicStateSpace; kwargs...) = psceig(psys.A; kwargs...)
"""
    pszero(psys::PeriodicStateSpece{HarmonicArray}[, N]; P, atol, rtol, fast) -> val

Compute the finite in infinite zeros of a continuous-time periodic system `psys = (Ahr(t), Bhr(t), Chr(t), Dhr(t))` in `val`, 
where the periodic system matrices `Ahr(t)`, `Bhr(t)`, `Chr(t)`, and `Dhr(t)` are in a _Harmonic Array_ representation. 
`N` is the number of selected harmonic components in the Fourier series of the system matrices (default: `N = max(20,nh-1)`, 
where `nh` is the maximum number of harmonics terms) and the keyword parameter `P` is the number of full periods 
to be considered (default: `P = 1`) to build 
a frequency-lifted LTI representation based on truncated block Toeplitz matrices (see [`ps2fls`](@ref)). 
"""
function pszero(psys::PeriodicStateSpace{PM}, N::Union{Int,Missing} = missing; P::Int= 1, fast::Bool = true, atol::Real = 0, rtol::Real = 0) where {PM <: HarmonicArray}
    ismissing(N) && (N = max(20, max(size(psys.A.values,3),size(psys.B.values,3),size(psys.C.values,3),size(psys.D.values,3))-1))
    (N == 0 || islti(psys) ) && (return spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])
    
    ωhp2 = pi/P/psys.A.period
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
       # Conjecture: number of infinite zeros is the same as that of the time-evaluated system 
       zm = spzeros(dssdata(psteval(psys, rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
       return [σf; zm[isinf.(zm)]]
    else
       return σf
    end
end
"""
    pszero(psys::PeriodicStateSpece{FourierFunctionMatrix}[, N]; P, atol, rtol, fast) -> val

Compute the finite in infinite zeros of a continuous-time periodic system `psys = (Af(t), Bf(t), Cf(t), Df(t))` in `val`, 
where the periodic system matrices `Af(t)`, `Bf(t)`, `Cf(t)`, and `Df(t)` are in a _Fourier Function Matrix_ representation. 
`N` is the number of selected harmonic components in the Fourier series of the system matrices (default: `N = max(20,nh-1)`, 
where `nh` is the maximum number of harmonics terms) and the keyword parameter `P` is the number of full periods 
to be considered (default: `P = 1`) to build 
a frequency-lifted LTI representation based on truncated block Toeplitz matrices (see [`ps2frls`](@ref)). 

"""
function pszero(psys::PeriodicStateSpace{PM}, N::Union{Int,Missing} = missing; P::Int= 1, fast::Bool = true, atol::Real = 0, rtol::Real = 0) where {PM <: FourierFunctionMatrix}
    ismissing(N) && (N = max(20, maximum(ncoefficients.(Matrix(psys.A.M))), maximum(ncoefficients.(Matrix(psys.B.M))),
                                   maximum(ncoefficients.(Matrix(psys.C.M))), maximum(ncoefficients.(Matrix(psys.A.M)))))
    (N == 0 || islti(psys) ) && (return spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])

    # employ heuristics to determine fix finite zeros by comparing two sets of computed zeros
    z = spzeros(dssdata(ps2frls(psys, N; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 

    ωhp2 = pi/P/psys.A.period
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
       # Conjecture: number of infinite zeros is the same as that of the time-evaluated system 
       zm = spzeros(dssdata(psteval(psys, rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
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
