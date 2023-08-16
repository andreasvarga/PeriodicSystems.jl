"""
    pclyap(A, C; K = 10, adj = false, solver, reltol, abstol, intpol, intpolmeth) -> X
    pclyap(A, C; K = 10, adj = false, solver, reltol, abstol) -> X

Solve the periodic Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t) , if adj = true.               

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods. 
Additionally `C` must be symmetric. 
The resulting symmetric periodic solution `X` has the type `PeriodicFunctionMatrix` and 
`X(t)` can be used to evaluate the value of `X` at time `t`. 
`X` has the period set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

The multiple-shooting method of [1] is employed to convert the (continuous-time) periodic differential Lyapunov equation 
into a discrete-time periodic Lyapunov equation satisfied by a multiple point generator of the solution. 
The keyword argument `K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 10`). 
If  `A` and `C` are of type `PeriodicSwitchingMatrix`, then `K` specifies the number of grid points used between two consecutive switching time values (default: `K = 1`).  
The multiple point periodic generator is computed  by solving the appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 
The resulting periodic generator is finally converted into a periodic function matrix which determines for a given `t` 
the function value `X(t)` by integrating the appropriate ODE from the nearest grid point value. 

To speedup function evaluations, interpolation based function evaluations can be used 
by setting the keyword argument `intpol = true` (default: `intpol = false`). 
In this case the interpolation method to be used can be specified via the keyword argument
`intpolmeth = meth`. The allowable values for `meth` are: `"constant"`, `"linear"`, `"quadratic"` and `"cubic"` (default) (see also [`ts2pfm`](@ref)).
Interpolation is not possible if `A` and `C` are of type `PeriodicSwitchingMatrix`. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) (see [`tvstm`](@ref)). 
Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function pclyap(A::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic")
   if intpol
      return convert(PeriodicFunctionMatrix,pgclyap(A, C, K;  adj, solver, reltol, abstol), method = intpolmeth)
   else
      W0 = pgclyap(A, C, K;  adj, solver, reltol, abstol)
      return PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol),A.period)
   end
end
pclyap(A::PeriodicFunctionMatrix, C::AbstractMatrix; kwargs...) = pclyap(A, PeriodicFunctionMatrix(C, A.period; nperiod = A.nperiod); kwargs...)
function pclyap(A::PeriodicSymbolicMatrix, C::PeriodicSymbolicMatrix; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic")
   At = convert(PeriodicFunctionMatrix,A)
   Ct = convert(PeriodicFunctionMatrix,C)
   if intpol
      return convert(PeriodicFunctionMatrix,pgclyap(At, Ct, K;  adj, solver, reltol, abstol), method = intpolmeth)
   else
      W0 = pgclyap(At, Ct, K;  adj, solver, reltol, abstol)
      PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, At, Ct; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
   #convert(PeriodicSymbolicMatrix, pgclyap(convert(PeriodicFunctionMatrix,A), convert(PeriodicFunctionMatrix,C), K;  adj, solver, reltol, abstol))
end
function pclyap(A::HarmonicArray, C::HarmonicArray; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic")
   if intpol
      return convert(PeriodicFunctionMatrix,pgclyap(A, C, K;  adj, solver, reltol, abstol), method = intpolmeth)
   else
      W0 = pgclyap(A, C, K;  adj, solver, reltol, abstol)
      PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
   #convert(HarmonicArray, pgclyap(A,  C, K;  adj, solver, reltol, abstol))
end
function pclyap(A::FourierFunctionMatrix, C::FourierFunctionMatrix; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic")
   if intpol
      return convert(PeriodicFunctionMatrix,pgclyap(A, C, K;  adj, solver, reltol, abstol), method = intpolmeth)
   else
      W0 = pgclyap(A, C, K;  adj, solver, reltol, abstol)
      PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
   #convert(FourierFunctionMatrix, pgclyap(A,  C, K;  adj, solver, reltol, abstol))
end
function pclyap(A::PeriodicTimeSeriesMatrix, C::PeriodicTimeSeriesMatrix; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic")
   if intpol
      return convert(PeriodicFunctionMatrix,pgclyap(A, C, K;  adj, solver, reltol, abstol), method = intpolmeth)
   else
      W0 = pgclyap(A, C, K;  adj, solver, reltol, abstol)
      PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
   # pgclyap(convert(HarmonicArray,A), convert(HarmonicArray,C), K;  adj, solver, reltol, abstol)
end
function pclyap(A::PeriodicSwitchingMatrix, C::PeriodicSwitchingMatrix; K::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7)
   W0 = pgclyap(A, C, K;  adj, solver, reltol, abstol)
   PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
end


for PM in (:PeriodicFunctionMatrix, :PeriodicSymbolicMatrix, :HarmonicArray, :FourierFunctionMatrix, :PeriodicTimeSeriesMatrix)
   @eval begin
      function prclyap(A::$PM, C::$PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic") 
         pclyap(A, C; K, adj = true, solver, reltol, abstol)
      end
      function prclyap(A::$PM, C::AbstractMatrix; kwargs...)
         prclyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
      end
      function pfclyap(A::$PM, C::$PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = false, intpolmeth = "cubic") 
         pclyap(A, C; K, adj = false, solver, reltol, abstol)
      end
      function pfclyap(A::$PM, C::AbstractMatrix; kwargs...)
         pfclyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
      end
   end
end
function prclyap(A::PM, C::PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where {PM <: PeriodicSwitchingMatrix}
   pclyap(A, C; K, adj = true, solver, reltol, abstol)
end
function prclyap(A::PM, C::AbstractMatrix; kwargs...) where {PM <: PeriodicSwitchingMatrix}
   prclyap(A, PM(C, A.period; nperiod = A.nperiod); kwargs...)
end
function pfclyap(A::PM, C::PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where {PM <: PeriodicSwitchingMatrix}
   pclyap(A, C; K, adj = false, solver, reltol, abstol)
end
function pfclyap(A::PM, C::AbstractMatrix; kwargs...) where {PM <: PeriodicSwitchingMatrix}
   pfclyap(A, PM(C, A.period; nperiod = A.nperiod); kwargs...)
end

"""
    pfclyap(A, C; K = 10, solver, reltol, abstol, intpol, intpolmeth) -> X
    pfclyap(A, C; K = 10, solver, reltol, abstol) -> X

Solve the periodic forward-time Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) .               

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

This function is merely an interface to [`pclyap`](@ref) (see this function for the description of keyword parameters). 
"""
pfclyap(A::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7)
"""
    prclyap(A, C; K = 10, solver, reltol, abstol, intpol, intpolmeth) -> X
    prclyap(A, C; K = 10, solver, reltol, abstol) -> X

Solve the periodic reverse-time Lyapunov differential equation

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t).               

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

This function is merely an interface to [`pclyap`](@ref) (see this function for the description of keyword parameters). 
"""
prclyap(A::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix; K::Int = 10, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7)

"""
    pgclyap(A, C[, K = 1]; adj = false, solver, reltol, abstol, dt) -> X

Compute periodic generators for the periodic Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t) , if adj = true.
    
The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. 
If `A` and `C` have the types `PeriodicFunctionMatrix`, `HarmonicArray`, `FourierFunctionMatrix` or `PeriodicTimeSeriesMatrix`, 
then the resulting `X` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A` and `C` are constant matrices
and `N = K` otherwise. 
If `A` and `C` have the type `PeriodicSwitchingMatrix`, then `X` is a collection of periodic generator matrices 
determined as a periodic switching matrix,
whose switching times are the unique switching times contained in the union of the switching times of `A` and `C`. 
If `K > 1`, a refined grid of `K` equidistant values is used for each two consecutive 
switching times in the union.      
The period of `X` is set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. Any component matrix of `X` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Lyapunov into a discrete-time periodic Lyapunov equation satisfied by 
the generator solution in the grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt' (default: `dt = 0`, only used if `solver = "symplectic"`) (see [`tvstm`](@ref)). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function pgclyap(A::PM1, C::PM2, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, dt = 0) where
      {PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix}, PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix}} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C)
   na = Int(round(period/A.period))
   nc = Int(round(period/C.period))
   nperiod = gcd(na*A.nperiod, nc*C.nperiod)
   n = size(A,1)
   Ts = period/K/nperiod
   solver == "symplectic" && dt == 0 && (dt = K >= 100 ? Ts : Ts*K/100/nperiod)
   
   if isconstant(A) && isconstant(C)
      X = adj ? lyapc(tpmeval(A,0)', tpmeval(C,0)) :  lyapc(tpmeval(A,0), tpmeval(C,0))
   else
      T = promote_type(eltype(A),eltype(C),Float64)
      T == Num && (T = Float64)
      Ka = isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Ad = Array{T,3}(undef, n, n, Ka) 
      Cd = Array{T,3}(undef, n, n, K) 
      Threads.@threads for i = 1:Ka
         @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
      end
      if adj
         Threads.@threads for i = K:-1:1
            @inbounds Cd[:,:,i]  = tvclyap(A, C, (i-1)*Ts, i*Ts; adj, solver, reltol, abstol, dt) 
         end
         X = pslyapd(Ad, Cd; adj)
      else
         Threads.@threads for i = 1:K
               @inbounds Cd[:,:,i]  = tvclyap(A, C, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol, dt) 
         end
         X = pslyapd(Ad, Cd; adj)
      end
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod)
end
function pgclyap(A::PM1, C::PM2, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, dt = 0) where
      {PM1 <: PeriodicSwitchingMatrix, PM2 <: PeriodicSwitchingMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C)
   na = round(Int,period/A.period)
   nc = round(Int,period/C.period)
   nperiod = gcd(na*A.nperiod, nc*C.nperiod)
   n = size(A,1)
   tsub = period/nperiod
   #solver == "symplectic" && dt == 0 && (dt = K >= 100 ? Ts : Ts*K/100/nperiod)

   ts = unique(sort([A.ts;C.ts]))
   Kc = length(ts)
  
   if isconstant(A) && isconstant(C)
      X = adj ? lyapc(tpmeval(A,0)', tpmeval(C,0)) :  lyapc(tpmeval(A,0), tpmeval(C,0))
      return PeriodicSwitchingMatrix([X], ts, period; nperiod)
   else
      T = promote_type(eltype(A),eltype(C),Float64)
      Ka = isconstant(A) ? 1 : Kc
      Kc1 = Kc*K
      @show T, n, Ka*K 
      Ad = Array{T,3}(undef, n, n, Ka*K) 
      Cd = Array{T,3}(undef, n, n, Kc1) 
      Threads.@threads for i = 1:Ka
          tf = i == Ka ? tsub : ts[i+1]
          k = (i-1)*K+1
          @inbounds Ad[:,:,k] = exp(tpmeval(A,ts[i])*(tf-ts[i])/K) 
          K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
      end
      if adj
         Threads.@threads for i = Kc:-1:1
            t0 = i == Kc ? tsub : ts[i+1]
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvclyap(A, C, ts[i], ts[i]+(t0-ts[i])/K; adj, solver, reltol, abstol, dt) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
         X = pslyapd(Ad, Cd; adj)
      else
         k = 1
         Threads.@threads for i = 1:Kc
            tf = i == Kc ? tsub : ts[i+1]
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvclyap(A, C, ts[i]+(tf-ts[i])/K, ts[i]; adj, solver, reltol, abstol, dt) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
         X = pslyapd(Ad, Cd; adj)
         # A1 = PeriodicArray(Ad,A.period; nperiod = A.nperiod)
         # C1 = PeriodicArray(Cd,C.period; nperiod = C.nperiod)
         # X1 = PeriodicArray(X,period; nperiod)
         # @show norm(A1*X1*A1'+C1-pmshift(X1))
      end
   end
   if K == 1
      return PeriodicSwitchingMatrix([X[:,:,i] for i in 1:size(X,3)], ts, period; nperiod)
   else
      tt = T[]
      for i = 1:Kc
          tf = i == Kc ? tsub : ts[i+1]
          Δ = (tf-ts[i])/K
          push!(tt,(ts[i] .+ collect(T,0:K-1)*Δ)...)
      end
      return PeriodicSwitchingMatrix([X[:,:,i] for i in 1:size(X,3)], tt, period; nperiod)
   end
end
function pgclyap(A::PM1, C::PM2, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, dt = 0) where
                {PM1 <: PeriodicTimeSeriesMatrix, PM2 <: PeriodicTimeSeriesMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C)
   na = round(Int,period/A.period)
   nc = round(Int,period/C.period)
   nperiod = gcd(na*A.nperiod, nc*C.nperiod)
   n = size(A,1)
   tsub = period/nperiod
   #solver == "symplectic" && dt == 0 && (dt = K >= 100 ? Ts : Ts*K/100/nperiod)
   
   if A.period == C.period 
      nperiod = gcd(A.nperiod,C.nperiod)
      ns = div(lcm(A.nperiod*length(A),C.nperiod*length(C)),nperiod)
      Δ = A.period/nperiod/ns
      δ = Δ/2
   else       
      Tsub = A.period/A.nperiod
      Tsub ≈ C.period/C.nperiod || error("periods or subperiods must be equal for addition")
      nperiod = lcm(A.nperiod,C.nperiod)
      period = Tsub*nperiod
      ns = lcm(length(A),length(C))
      Δ = Tsub/ns
      δ = Δ/2
   end 
   Kc = ns
   Tsd = Δ/K
   Ts = Δ
   δ = Tsd/2
   if isconstant(A) && isconstant(C)
      X = adj ? lyapc(tpmeval(A,0)', tpmeval(C,0)) :  lyapc(tpmeval(A,0), tpmeval(C,0))
      return PeriodicTimeSeriesMatrix([X], period; nperiod)
   else
      T = promote_type(eltype(A),eltype(C),Float64)
      Ka = isconstant(A) ? 1 : Kc
      Kc1 = Kc*K
      Ad = Array{T,3}(undef, n, n, Ka*K) 
      Cd = Array{T,3}(undef, n, n, Kc1) 
      Threads.@threads for i = 1:Ka
          k = (i-1)*K+1
          @inbounds Ad[:,:,k] = exp(tpmeval(A,(i-1)*Ts+δ)*Tsd) 
          K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
      end
      if adj
         k = Kc1
         Threads.@threads for i = Kc:-1:1
            tf = (i-1)*Ts
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvclyap(A, C, tf, tf+Tsd; adj, solver, reltol, abstol, dt) 
            #K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k-K+1:k-1]  
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
         X = pslyapd(Ad, Cd; adj)
      else
         Threads.@threads for i = 1:Kc
            t0 = (i-1)*Ts
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvclyap(A, C, t0+Tsd, t0; adj, solver, reltol, abstol, dt) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
         X = pslyapd(Ad, Cd; adj)
         # A1 = PeriodicArray(Ad,A.period; nperiod = A.nperiod)
         # C1 = PeriodicArray(Cd,C.period; nperiod = C.nperiod)
         # X1 = PeriodicArray(X,period; nperiod)
         # @show norm(A1*X1*A1'+C1-pmshift(X1))
      end
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod)
end
function tvclyap_eval(t::Real,X::PeriodicTimeSeriesMatrix,A::PM1, C::PM2; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, dt = 0) where
   {PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicTimeSeriesMatrix}, PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicTimeSeriesMatrix}} 
   """
       tvclyap_eval(t, X, A, C; adj = false, solver, reltol, abstol, dt) -> Xval

   Compute the time value `Xval := X(t)` of the solution of the periodic Lyapunov differential equation

      .
      W(t) = A(t)W(t) + W(t)A(t)' + C(t) ,  W(t0) = X(t0), t > t0, if adj = false

   or 

      .
     -W(t) = A(t)'W(t) + W(t)A(t) + C(t) ,  W(t0) = X(t0), t < t0, if adj = true,  

   using the periodic generator `X` determined with the function [`pgclyap`](@ref)). 
   The initial time `t0` is the nearest time grid value to `t`, from below, if `adj = false`, or from above, if `adj = true`. 
  
   The above ODE is solved by employing the integration method specified via the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt' (default: `dt = 0`, only used if `solver = "symplectic"`) (see [`tvstm`](@ref)). 
   """
   tsub = X.period/X.nperiod
   ns = length(X.values)
   Δ = tsub/ns
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
   if adj 
      ind = round(Int,tf/Δ)
      if ind == ns
         t0 = ind*Δ; ind = 1
      else
         t0 = (ind+1)*Δ; ind = ind+2; 
         ind > ns && (ind = 1) 
     end 
   else
      ind = round(Int,tf/Δ)
      ind == 0 && (ind = 1) 
      t0 = (ind-1)*Δ
   end
   return tvclyap(A, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol, dt) 
end
function tvclyap_eval(t::Real,X::PeriodicSwitchingMatrix,A::PM1, C::PM2; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, dt = 0) where
                     {PM1 <: PeriodicSwitchingMatrix, PM2 <: PeriodicSwitchingMatrix}  
   tsub = X.period/X.nperiod
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
   if adj 
      ind = findfirst(X.ts .> tf*(1+10*eps()))
      isnothing(ind) ? (ind = 1; t0 = tsub) : t0 = X.ts[ind]; 
   else
      ind = findfirst(X.ts .> tf*(1+10*eps()))
      isnothing(ind) ? ind = length(X) : ind -= 1
      t0 = X.ts[ind]
   end
   return tvclyap(A, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol, dt) 
end

function tvclyap(A::PM1, C::PM2, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where
   {T1, T2, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}} 
   #{PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}, PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}} 
   """
      tvclyap(A, C, tf, t0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at tf > t0 of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(t0) = 0, tf > t0, if adj = false

   or 
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(t0) = 0, tf < t0, if adj = true. 

   The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt' (default: `dt = abs(tf-t0)/100`, only used if `solver = "symplectic"`) (see [`tvstm`](@ref)). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   ismissing(W0) ? u0 = zeros(T,div(n*(n+1),2)) : u0 = MatrixEquations.triu2vec(W0)
   tspan = (T(t0),T(tf))
   fclyap!(du,u,p,t) = adj ? muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
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
   return MatrixEquations.vec2triu(sol[end], her=true)     
end
function muladdcsym!(y::AbstractVector, x::AbstractVector, isig, A::AbstractMatrix, C::AbstractMatrix)
   # A*X + X*A' + C
   n = size(A, 1)
   T1 = promote_type(eltype(A), eltype(x))
   # TO DO: eliminate building of X by using directly x
   X = MatrixEquations.vec2triu(convert(AbstractVector{T1}, x), her=true)
   @inbounds begin
      k = 1
      for j = 1:n
         for i = 1:j
            temp = C[i,j]
            for l = 1:n
               temp += A[i,l] * X[l,j] + X[i,l] * A[j,l]
            end
            y[k] = isig*temp
            k += 1
         end
      end
   end
   return y
end
