"""
     psc2d([PMT,] psysc, Ts; solver, reltol, abstol, dt) -> psys::PeriodicStateSpace{PMT}

Compute for the continuous-time periodic system `psysc = (A(t),B(t),C(t),D(t))` of period `T` and 
for a sampling time `Ts`, the corresponding discretized
periodic system `psys = (Ad,Bd,Cd,Dd)` using a zero-order hold based discretization method. 
The resulting discretized system `psys` has the matrices of type `PeriodicArray` by default, or
of type `PMT`, where `PMT` is one of the types `PeriodicMatrix`, `PeriodicArray`, `SwitchingPeriodicMatrix`
or `SwitchingPeriodicArray`.    

The discretization is performed by determining the monodromy matrix as a product of 
`K = T/Ts` state transition matrices of the extended state-space matrix `[A(t) B(t); 0 0]` 
by integrating numerically the corresponding homogeneous linear ODE.  
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = Ts/10`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "linear"` - use a special solver for linear ODEs (`MagnusGL6()`) with fixed time step `dt`;

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

For large values of `K`, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  
"""
function PeriodicSystems.psc2d(psysc::PeriodicStateSpace{PM}, Ts::Real; solver::String  = "auto", reltol = 1e-3, abstol = 1e-7, dt = Ts/10) where {PM <: FourierFunctionMatrix}
    Ts > 0 || error("the sampling time Ts must be positive")
    period = psysc.period
    r = rationalize(period/Ts)
    denominator(r) == 1 || error("incommensurate period and sample time")
    K = numerator(r)

    T = eltype(psysc)
    T1 = T <: BlasFloat ? T : (T <: Num ? Float64 : promote_type(Float64,T)) 
    n, m = size(psysc.B); p = size(psysc.D,1)
    PMT = PeriodicArray


    # quick exit in constant case  
    islti(psysc) && (return ps(PMT,c2d(psaverage(psysc),Ts)[1],psysc.period))

    kk = gcd(K,psysc.A.nperiod)
    ka, na = PeriodicMatrices.isconstant(psysc.A) ? (1,K) :  (div(K,kk),kk)
    kk = gcd(K,psysc.B.nperiod)
    (kb, nb) = PeriodicMatrices.isconstant(psysc.B) ? (1,K) :  (div(K,kk),kk)
    kk = gcd(K,psysc.C.nperiod)
    kc, nc = PeriodicMatrices.isconstant(psysc.C) ? (1,K) : (div(K,kk),kk)
    kk = gcd(K,psysc.D.nperiod)
    kd, nd = PeriodicMatrices.isconstant(psysc.D) ? (1,K) : (div(K,kk),kk)
    Ap = Array{T,3}(undef,n,n,ka)
    Bp = Array{T,3}(undef,n,m,kb)
    Cp = Array{T,3}(undef,p,n,kc)
    Dp = Array{T,3}(undef,p,m,kd)

    i1 = 1:n; i2 = n+1:n+m
    if PeriodicMatrices.isconstant(psysc.A) && PeriodicMatrices.isconstant(psysc.B)
        G = exp([ rmul!(tpmeval(psysc.A,0),Ts) rmul!(tpmeval(psysc.B,0),Ts); zeros(T1,m,n+m)])
        Ap = view(G,i1,i1)
        Bp = view(G,i1,i2)
    else
        kab = max(ka,kb)
        Gfun = PeriodicFunctionMatrix(t -> [tpmeval(psysc.A,t) tpmeval(psysc.B,t); zeros(T1,m,n+m)], period; nperiod = div(K,kab))
        #G = monodromy(Gfun, kab; Ts, solver, reltol, abstol, dt)
        G = monodromy(Gfun, kab; solver, reltol, abstol, dt)
        Ap = view(G.M,i1,i1,1:ka)
        Bp = view(G.M,i1,i2,1:kb)
    end
    [copyto!(view(Cp,:,:,i),tpmeval(psysc.C,(i-1)*Ts)) for i in 1:kc]
    [copyto!(view(Dp,:,:,i),tpmeval(psysc.D,(i-1)*Ts)) for i in 1:kd]
    return ps(PMT(Ap,period; nperiod = na),PMT(Bp,period; nperiod = nb),PMT(Cp,period; nperiod = nc),PMT(Dp,period; nperiod = nd))
    # end PSC2D
end
function PeriodicSystems.psc2d(PMT::Type, psysc::PeriodicStateSpace{PM}, Ts::Real; kwargs...) where {PM <: FourierFunctionMatrix} 
    PMT ∈ (PeriodicMatrix, PeriodicArray, SwitchingPeriodicMatrix, SwitchingPeriodicArray) ||
           error("only discrete periodic matrix types allowed")
    convert(PeriodicStateSpace{PMT}, psc2d(psysc, Ts; kwargs...))
end
