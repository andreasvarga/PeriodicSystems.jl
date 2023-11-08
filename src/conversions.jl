"""
    psaverage(psysc) -> sys::DescriptorStateSpace

Compute for the continuous-time periodic system `psysc = (A(t),B(t),C(t),D(t))` 
the corresponding time averaged LTI system `sys = (Am,Bm,Cm,Dm)` over one period.  
"""
function psaverage(psysc::PeriodicStateSpace{PM}) where {T,PM <: AbstractPeriodicArray{:c,T}}
    return dss(pmaverage(psysc.A), pmaverage(psysc.B), pmaverage(psysc.C), pmaverage(psysc.D))
end
"""
    psteval(psys,tval) -> sys::DescriptorStateSpace

Compute for the periodic system `psys = (A(t),B(t),C(t),D(t))` and a time value `tval`, 
the LTI system `sys = (A(tval),B(tval),C(tval),D(tval))`. If `A(tval)` is not square, then 
`A(tval)` is padded with zeros to form a square matrix and appropriate numbers of zero rows and zero columns are added to 
`B(tval)` and `C(tval)`, respectively. 
"""
function psteval(psys::PeriodicStateSpace{PM}, tval::Real = 0) where {PM <: AbstractPeriodicArray}
    A = tpmeval(psys.A, tval)
    B = tpmeval(psys.B, tval)
    C = tpmeval(psys.C, tval)
    n1, n2 = size(A)
    if n1 == n2
       return dss(A, B, C, tpmeval(psys.D, tval); Ts = psys.Ts )
    else
       n = max(n1,n2) 
       T = eltype(A)
       return dss([A zeros(T,n1,n-n2);zeros(T,n-n1,n)], [B; zeros(T,n-n1,size(B,2))], [C zeros(T,size(C,1),n-n2)], tpmeval(psys.D, tval); Ts = psys.Ts )
    end
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
`K = T/Ts` state transition matrices of the extended state-space matrix `[A(t) B(t); 0 0]` 
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
function psc2d(psysc::PeriodicStateSpace{PM}, Ts::Real; solver::String  = "", reltol = 1e-3, abstol = 1e-7, dt = Ts/10) where {PM <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix}}

    Ts > 0 || error("the sampling time Ts must be positive")
    period = psysc.period
    r = rationalize(period/Ts)
    denominator(r) == 1 || error("incommensurate period and sample time")
    K = numerator(r)

    T = eltype(psysc)
    T1 = T <: BlasFloat ? T : (T <: Num ? Float64 : promote_type(Float64,T)) 
    n, m = size(psysc.B) 

    # quick exit in constant case  
    islti(psysc) && (return ps(c2d(psaverage(psysc),Ts)[1],psysc.period))

    kk = gcd(K,psysc.A.nperiod)
    ka, na = isconstant(psysc.A) ? (1,K) :  (div(K,kk),kk)
    kk = gcd(K,psysc.B.nperiod)
    (kb, nb) = isconstant(psysc.B) ? (1,K) :  (div(K,kk),kk)
    kk = gcd(K,psysc.C.nperiod)
    kc, nc = isconstant(psysc.C) ? (1,K) : (div(K,kk),kk)
    kk = gcd(K,psysc.D.nperiod)
    kd, nd = isconstant(psysc.D) ? (1,K) : (div(K,kk),kk)
    Ap = similar(Vector{Matrix},ka)
    Bp = similar(Vector{Matrix},kb)
    Cp = similar(Vector{Matrix},kc)
    Dp = similar(Vector{Matrix},kd)

    i1 = 1:n; i2 = n+1:n+m
    if isconstant(psysc.A) && isconstant(psysc.B)
        G = exp([ rmul!(tpmeval(psysc.A,0),Ts) rmul!(tpmeval(psysc.B,0),Ts); zeros(T1,m,n+m)])
        Ap[1] = G[i1,i1]
        Bp[1] = G[i1,i2]
        [Cp[i] = tpmeval(psysc.C,(i-1)*Ts) for i in 1:kc]
        [Dp[i] = tpmeval(psysc.D,(i-1)*Ts) for i in 1:kd]
    else
        kab = max(ka,kb)
        Gfun = PeriodicFunctionMatrix(t -> [tpmeval(psysc.A,t) tpmeval(psysc.B,t); zeros(T1,m,n+m)], period; nperiod = div(K,kab))
        #G = monodromy(Gfun, kab; Ts, solver, reltol, abstol, dt)
        G = monodromy(Gfun, kab; solver, reltol, abstol, dt)
        [Ap[i] = G.M[i1,i1,i] for i in 1:ka]
        [Bp[i] = G.M[i1,i2,i] for i in 1:kb]
        [Cp[i] = tpmeval(psysc.C,(i-1)*Ts) for i in 1:kc]
        [Dp[i] = tpmeval(psysc.D,(i-1)*Ts) for i in 1:kd]
    end
    PMT = PeriodicMatrix
    return ps(PMT(Ap,period; nperiod = na),PMT(Bp,period; nperiod = nb),PMT(Cp,period; nperiod = nc),PMT(Dp,period; nperiod = nd))
    # end PSC2D
end
# psc2d(psysc::PeriodicStateSpace{PeriodicTimeSeriesMatrix{:c,T}}, Ts::Real; kwargs...) where {T} = 
#       psc2d(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psysc), Ts; kwargs...)
psc2d(psysc::PeriodicStateSpace{PeriodicTimeSeriesMatrix{:c,T}}, Ts::Real; kwargs...) where {T} = 
      psc2d(convert(PeriodicStateSpace{HarmonicArray},psysc), Ts; kwargs...)
psc2d(psysc::PeriodicStateSpace{PeriodicSymbolicMatrix{:c,T}}, Ts::Real; kwargs...) where {T} = 
      psc2d(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psysc), Ts; kwargs...)
