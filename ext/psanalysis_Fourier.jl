PeriodicSystems.pspole(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, N::Int = 10; kwargs...)  = psceigfr(psys.A, N; kwargs...)
"""
    pszero(psys::PeriodicStateSpece{FourierFunctionMatrix}[, N]; P, atol, rtol, fast) -> val

Compute the finite and infinite zeros of a continuous-time periodic system `psys = (Af(t), Bf(t), Cf(t), Df(t))` in `val`, 
where the periodic system matrices `Af(t)`, `Bf(t)`, `Cf(t)`, and `Df(t)` are in a _Fourier Function Matrix_ representation. 
`N` is the number of selected harmonic components in the Fourier series of the system matrices (default: `N = max(20,nh-1)`, 
where `nh` is the maximum number of harmonics terms) and the keyword parameter `P` is the number of full periods 
to be considered (default: `P = 1`) to build 
a frequency-lifted LTI representation based on truncated block Toeplitz matrices. 

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
function PeriodicSystems.pszero(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, N::Union{Int,Missing} = missing; P::Int= 1, fast::Bool = true, atol::Real = 0, rtol::Real = 0) 
    ismissing(N) && (N = max(20, maximum(ncoefficients.(Matrix(psys.A.M))), maximum(ncoefficients.(Matrix(psys.B.M))),
                                   maximum(ncoefficients.(Matrix(psys.C.M))), maximum(ncoefficients.(Matrix(psys.A.M)))))
    (N == 0 || islti(psys) ) && (return MatrixPencils.spzeros(dssdata(psaverage(psys))...; fast, atol1 = atol, atol2 = atol, rtol)[1])

    # employ heuristics to determine fix finite zeros by comparing two sets of computed zeros
    z = MatrixPencils.spzeros(dssdata(ps2frls(psys, N; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 

    period = psys.A.period
    ωhp2 = pi/P/period
    n = size(psys.A,1)
    T = promote_type(Float64, eltype(psys.A))
    zf = z[isfinite.(z)]
    ind = sortperm(imag(zf),by=abs); 
    nf = count(abs.(imag(zf[ind[1:min(4*n,length(ind))]])) .<=  ωhp2*(1+sqrt(eps(T))))
    zf = zf[ind[1:nf]]

    z2 = MatrixPencils.spzeros(dssdata(ps2frls(psys, N+2; P))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
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
       zm = MatrixPencils.spzeros(dssdata(psteval(psys, period*rand()))...; fast, atol1 = atol, atol2 = atol, rtol)[1] 
       zf = [σf; zm[isinf.(zm)]]
    end
    nz = length(zf)
    nz > n && (@warn "$(nz-n) spurious finite zero(s) present")
    return zf
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

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

_References_

[1] P. Colaneri. Continuous-time periodic systems in H2 and H∞: Part I: Theoretical Aspects.
    Kybernetika, 36:211-242, 2000. 

[2] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function PeriodicSystems.psh2norm(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, K::Int = 1; adj::Bool = false, smarg::Real = 1, fast::Bool = false, 
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
    return sqrt(sum(μ)*P.nperiod/psys.period)
end
function PeriodicSystems.tvh2norm(A::PM1, B::PM2, C::PM3, P::AbstractMatrix, tf, t0; adj = false, solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where
    {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
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
    absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt` (default: `dt = abs(tf-t0)/100`, only used if `solver = "symplectic"`) 
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

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 


_References_

[1] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""
function PeriodicSystems.pshanorm(psys::PeriodicStateSpace{<: FourierFunctionMatrix}, K::Int = 1; smarg::Real = 1, 
                  offset::Real = sqrt(eps()), solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) 
    !isstable(psys, K; smarg, offset, solver, reltol, abstol) && error("The system must be stable")  # unstable system
    Q = pgclyap(psys.A, psys.C'*psys.C, K; adj = true, solver, reltol, abstol) 
    P = pgclyap(psys.A, psys.B*psys.B', K; adj = false, solver, reltol, abstol)
    return sqrt(maximum(norm.(eigvals(P*Q),Inf)))
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

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 


_References:_    

[1] P. Colaneri. Continuous-time periodic systems in H2 and H∞: Part I: Theoretical Aspects.
    Kybernetika, 36:211-242, 2000. 

[2] A. Varga, On solving periodic differential matrix equations with applications to periodic system norms computation.
    Proc. CDC/ECC, Seville, p.6545-6550, 2005.  
"""   
function PeriodicSystems.pslinfnorm(psys::PeriodicStateSpace{<: FourierFunctionMatrix{:c,T}}, K::Int=100; hinfnorm::Bool = false, rtolinf::Real = float(real(T))(0.001), fast::Bool = true, 
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
          PeriodicSystems.checkham1(psys,g,K,toluc1, toluc2, solver, reltol, abstol, dt) ? gl = g : gu = g   
          g = (gl+gu)/2
    end
    return g, nothing
end
function PeriodicSystems.checkham1(psys::PeriodicStateSpace{<: FourierFunctionMatrix{:c,T}}, g::Real, K::Int,toluc1, toluc2, solver, reltol, abstol, dt)  where {T}
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
PeriodicSystems.checkham1(psys::PeriodicStateSpace{<: FourierFunctionMatrix{:c,T}}, g::Real, K::Int)  where {T} = PeriodicSystems.checkham1(psys,g,K,eps(), sqrt(eps()), "symplectic", 1.e-10,1.e-10,0)
function PeriodicSystems.pshinfnorm(psys::PeriodicStateSpace{<: FourierFunctionMatrix{:c,T}}; 
    rtolinf::Real = float(real(T))(0.001), offset::Real = sqrt(eps(float(real(T)))), solver = "symplectic", reltol = 1e-6, abstol = 1e-7, dt = 0)  where {T} 
    return pslinfnorm(psys; hinfnorm = true, rtolinf, solver, reltol, abstol, dt)
end
