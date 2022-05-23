"""
    psaverage(psysc) -> sys::DescriptorStateSpace

Compute for the continuous-time periodic system `psysc = (A(t),B(t),C(t),D(t))` 
the corresponding time averaged LTI system `sys = (Am,Bm,Cm,Dm)` over one period.  
"""
function psaverage(psysc::PeriodicStateSpace{PM}) where {T,PM <: AbstractPeriodicArray{:c,T}}
    return dss(pmaverage(psysc.A), pmaverage(psysc.B), pmaverage(psysc.C), pmaverage(psysc.D))
end
"""
    psteval(psysc,tval) -> sys::DescriptorStateSpace

Compute for the continuous-time periodic system `psysc = (A(t),B(t),C(t),D(t))` and a time value `tval`, 
the LTI system `sys = (A(tval),B(tval),C(tval),D(tval))`.  
"""
function psteval(psysc::PeriodicStateSpace{PM}, tval::Real = 0) where {T,PM <: AbstractPeriodicArray{:c,T}}
    return dss(tvmeval(psysc.A, tval)[1], tvmeval(psysc.B, tval)[1], tvmeval(psysc.C, tval)[1], tvmeval(psysc.D, tval)[1])
end
"""
     psmrc2d(sys, Ts; ki, ko) -> psys::PeriodicStateSpace{PeriodicMatrix}

Perform the multirate discretization of a linear time-invariant system.

For a continuous-time state-space system `sys = (A,B,C,D)`, a basic sampling time `Ts`, 
and the integer vectors `ki` and `ko` containing, respectively, the numbers of 
input and output sampling subperiods, the corresponding discretized 
periodic system `psys = (Ap,Bp,Cp,Dp)` of period `T = n*Ts` is determined, 
where `n` is the least common multiple
of the integers contained in `ki` and `ko`. For a continuous-time system `sys` 
a zero-order hold based discretization method is used, such that the `i`-th input 
is constant during intervals of length `ki[i]*Ts`. 
An output hold device is used to provide constant intersample outputs, such that the `i`-th output
is constant during intervals of length `ko[i]*Ts`. 
For a discrete-time system with a defined sample time `Δ`, 
an input and output resampling is performed using `Δ` as basic  
sample time and the value of `Ts` is ignored. 
If the system sample time is not defined, then the value of
`Ts` is used as the basic sample time. 
"""
function psmrc2d(sys::DescriptorStateSpace{T}, Ts::Real; ki::Vector{Int} = ones(Int,size(sys,2)), ko::Vector{Int}= ones(Int,size(sys,1))) where T
    Ts > 0 || error("the sampling time Ts must be positive")
    (all(ki .>= 1) && all(ko .>= 1)) || error("the numbers of subperiods must be at least one")
    p, m = size(sys)
    length(ki) == m || error("the number of input components of ki must be $m")
    length(ko) == p || error("the number of ouput components of ko must be $p")
    A, E, B, C, D = dssdata(sys)
    E == I || error("only standard state-space systems supported")
    Δ = sys.Ts
    T1 = T <: BlasFloat ? T : promote_type(Float64,T)
    n = sys.nx
    if Δ != 0
       i1 = 1:n; i2 = n+1:n+m
       Δ > 0 && (Ts = Δ)
       G = exp([ rmul!(A,Ts) rmul!(B,Ts); zeros(T1,m,n+m)])
       A =  G[i1,i1]
       B =  G[i1,i2]
    end

    K = lcm(lcm(ki),lcm(ko))
    Ap = similar(Vector{Matrix},K)
    Bp = similar(Vector{Matrix},K)
    Cp = similar(Vector{Matrix},K)
    Dp = similar(Vector{Matrix},K)
    for i = 1:K
        si = mod.(i-1,ki) .== 0
        Si = Diagonal(si)
        Sineg = Diagonal(.!si)
        Ti = Diagonal(mod.(i-1,ko) .== 0)
        Ap[i] = [A  B*Sineg; zeros(T1,m,n) Sineg]
        Bp[i] = [B*Si; Si]
        Cp[i] = [Ti*C Ti*D*Sineg]
        Dp[i] = Ti*D
    end
    PMT = PeriodicMatrix
    period = K*Ts
    return ps(PMT(Ap,period),PMT(Bp,period),PMT(Cp,period),PMT(Dp,period))
end
"""
     psc2d(psysc, Ts; solver, reltol, abstol, dt) -> psys::PeriodicStateSpace{PeriodicMatrix}

Compute for the continuous-time periodic system `psysc = (A(t),B(t),C(t),D(t))` of period `T` and 
for a sampling time `Ts`, the corresponding discretized
periodic system `psys = (Ad,Bd,Cd,Dd)` using a zero-order hold based discretization method. 

The discretization is performed by determining the monodromy matrix as a product of 
`K = T/Ts` state transition matrices of the extended state-space matrix [A(t) B(t); 0 0] 
by integrating numerically the corresponding homogeneous linear ODE.  
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-3`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = Ts/10`) (see [`tvstm`](@ref)). 
For large values of `K`, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  
"""
function psc2d(psysc::PeriodicStateSpace{PM}, Ts::Real; solver::String  = "", reltol = 1e-3, abstol = 1e-7, dt = Ts/10) where {T,PM <: AbstractPeriodicArray{:c,T}}

    Ts > 0 || error("the sampling time Ts must be positive")
    period = psysc.period
    r = rationalize(period/Ts)
    denominator(r) == 1 || error("incommensurate period and sample time")
    K = numerator(r)

    T1 = T <: BlasFloat ? T : (T <: Num ? Float64 : promote_type(Float64,T)) 
    ONE = one(T1)
    n, m = size(psysc.B) 

    # quick exit in constant case  
    islti(psysc) && (return ps(c2d(psaverage(psysc),Ts)[1],psysc.period))

    ka, na = isconstant(psysc.A) ? (1,K) : (K,1)
    kb, nb = isconstant(psysc.A) && isconstant(psysc.B) ? (1,K) : (K,1)
    kc, nc = isconstant(psysc.C) ? (1,K) : (K,1)
    kd, nd = isconstant(psysc.D) ? (1,K) : (K,1)
    Ap = similar(Vector{Matrix},ka)
    Bp = similar(Vector{Matrix},kb)
    Cp = similar(Vector{Matrix},kc)
    Dp = similar(Vector{Matrix},kd)

    psys1 = typeof(psysc.A) <: PeriodicFunctionMatrix ? psysc : 
               convert(PeriodicStateSpace{PeriodicFunctionMatrix},psysc)
    i1 = 1:n; i2 = n+1:n+m

    if isconstant(psysc.A) && isconstant(psysc.B)
        G = exp([ rmul!(psys1.A.f(0),Ts) rmul!(psys1.B.f(0),Ts); zeros(T1,m,n+m)])
        Ap[1] = G[i1,i1]
        Bp[1] = G[i1,i2]
        ts = zero(T1)
        for i = 1:kc
            Cp[i] = psys1.C.f(ts)
            ts += Ts
        end
        ts = zero(T1)
        for i = 1:kd
            Dp[i] = psys1.D.f(ts)
            ts += Ts
        end
    else
        Gfun = PeriodicFunctionMatrix(t -> [psys1.A.f(t) psys1.B.f(t); zeros(T1,m,n+m)], period)
        G = monodromy(Gfun, K; solver, reltol, abstol, dt)
        [Ap[i] = G.M[i1,i1,i] for i in 1:ka]
        [Bp[i] = G.M[i1,i2,i] for i in 1:kb]
        ts = zero(T1)
        for i = 1:kc
            Cp[i] = psys1.C.f(ts)
            ts += Ts
        end
        ts = zero(T1)
        for i = 1:kd
            Dp[i] = psys1.D.f(ts)
            ts += Ts
        end
    end
    PMT = PeriodicMatrix
    return ps(PMT(Ap,period; nperiod = na),PMT(Bp,period; nperiod = nb),PMT(Cp,period; nperiod = nc),PMT(Dp,period; nperiod = nd))
    # end PSC2D
end
