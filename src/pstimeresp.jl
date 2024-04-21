"""
    pstimeresp(sys, u, t, x0 = zeros(sys.nx); state_history = false, solver, reltol, abstol) -> (y, tout, x)

Compute the time response of a periodic system `sys = (A(t),B(t),C(t),D(t))` to the input signals 
described by `u` and `t`. The time vector `t` consists of regularly spaced time samples. 
The input `u` can be specified as a matrix with as many columns as the number of inputs of `sys`, 
in which case its `i`-th row specifies the input values at time `t[i]`. 
For a discrete-time system, `u` should be sampled at the same sampling rate `Ts` as `sys`
and `t` must have all time steps equal to `Ts` or can be set to an empty vector. 
For continuous-time models, the input values are interpolated between samples using  
zero-order hold based interpolation. 
The vector `x0` specifies the initial state vector at time `t[1]` and is set to zero when omitted. 

The matrix `y` contains the resulting time history of the outputs of `sys` and 
the vector `tout` contains the corresponding values of the time samples.
The `i`-th row of `y` contains the output values at time `tout[i]`.  
If the keyword parameter value `state_history = true` is used, then the matrix `x` contains 
the resulting time history of the state vector and its `i`-th row `x[i,:]` contains 
the state values at time `tout[i]`. 
The column dimension of the matrix `x` is `n`, the dimension of the state vector. 
For a discrete-time periodic system with time-varying state dimensions, `n` is the maximum value of the 
dimensions of the state vector over one period. The components of `x[i,:]` have trailing zero values if the 
corresponding state vector dimension is less than `n`.  
By default, the state history is not saved and `x = nothing`.


For a continuous-time model an equivalent discretized model is determined to be used for simulation.
The discretization is performed by determining the monodromy matrix as a product of 
state transition matrices of the extended state-space matrix `[A(t) B(t); 0 0]` 
by integrating numerically the corresponding homogeneous linear ODE.  
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = Ts/10`) (see [`tvstm`](@ref)). 
For large numbers of product terms, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable. 
"""
function pstimeresp(psys::PeriodicStateSpace{PM}, u::AbstractVecOrMat{<:Number}, t::AbstractVector{<:Real},  
                  x0::AbstractVector{<:Number} = zeros(T,psys.nx[1]); 
                  state_history::Bool = false, solver::String  = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where {Domain,T, PM <: AbstractPeriodicArray{Domain,T}}

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   n = psys.nx[1] 
   p, m = psys.ny[1], psys.nu[1]

   N, m1 = size(u,1), size(u,2)
   m == m1 || error("u must have as many columns as system inputs")
   n == length(x0) || error("x0 must have the same length as the system state vector")

   disc = isdiscrete(psys)
   Ts = psys.Ts
   ns = length(t)
   ns > 0 && ns != N && error("u must have the same number of rows as the number of values in t")
   if ns > 1
      Δ = t[2]-t[1]
      disc && Ts > 0 && abs.(Ts-Δ) > 0.0000001*Δ 
      (any(diff(t) .<= 0) || any(isinf.(t)) || any(abs.(diff(t).-Δ).> 0.00001*Δ)) && 
           error("time vector t must contain monotonically increasing and evenly spaced time samples")
      tout = t
   else
      if disc
         Δ = abs(Ts)           
         tout = Vector{real(T1)}(0:Δ:(N-1)*Δ) 
      else
         error("Time values must be given in the continuous-time case")
      end       
   end 
   p == 1 ? y = Vector{T1}(undef, N) : y = Matrix{T1}(undef, N, p) 
   xt = copy(x0)
   if disc
      pa = length(psys.A) 
      pb = length(psys.B)
      pc = length(psys.C) 
      pd = length(psys.D)
      if PM <: PeriodicArray
         state_history ?  x = Matrix{T1}(undef, N, n) : x = nothing 
         for i = 1:N 
             ia = mod(i-1,pa)+1
             ib = mod(i-1,pb)+1
             ic = mod(i-1,pc)+1
             id = mod(i-1,pd)+1
             ut = u[i,:]
             y[i,:] = psys.C.M[:,:,ic]*xt + psys.D.M[:,:,id]*ut
             state_history && (x[i,:] = xt) 
             xt = psys.A.M[:,:,ia]*xt + psys.B.M[:,:,ib]*ut
         end
      elseif PM <: PeriodicMatrix
         nmax = maximum(psys.nx)
         state_history ?  x = zeros(T1, N, nmax) : x = nothing 
         for i = 1:N 
             ia = mod(i-1,pa)+1
             ib = mod(i-1,pb)+1
             ic = mod(i-1,pc)+1
             id = mod(i-1,pd)+1
             ut = u[i,:]
             y[i,:] = psys.C.M[ic]*xt + psys.D.M[id]*ut
             state_history && (copyto!(view(x,i,1:length(xt)), xt)) 
             xt = psys.A.M[ia]*xt + psys.B.M[ib]*ut
         end
      else
         state_history ?  x = zeros(T1, N, nmax) : x = nothing 
         for i = 1:N 
             ut = u[i,:]
             y[i,:] = kpmeval(psys.C,i)*xt + kpmeval(psys.D,i)*ut
             state_history && (copyto!(view(x,i,1:length(xt)), xt)) 
             xt = kpmeval(psys.A,i)*xt + kpmeval(psys.B,i)*ut
         end
      end
      return y, tout, x
   else
      # for commensurate time-steps and final time larger than one period use discretized model
      if rationalize(psys.period/Δ).den == 1 && (tout[end] ≥ psys.period || tout[end] ≈ psys.period)
         state_history ?  x = Matrix{T1}(undef, N, n) : x = nothing 
         dt == 0 && (dt = Δ/10)
         psysd = psc2d(psys, Δ; solver, reltol, abstol, dt)
         pa = length(psysd.A) 
         pb = length(psysd.B)
         pc = length(psysd.C) 
         pd = length(psysd.D)
         for i = 1:N 
             ia = mod(i-1,pa)+1
             ib = mod(i-1,pb)+1
             ic = mod(i-1,pc)+1
             id = mod(i-1,pd)+1
             ut = u[i,:]
            #  y[i,:] = psysd.C.M[ic]*xt + psysd.D.M[id]*ut
            #  state_history && (x[i,:] = xt) 
            #  xt = psysd.A.M[ia]*xt + psysd.B.M[ib]*ut
             y[i,:] = psysd.C.M[:,:,ic]*xt + psysd.D.M[:,:,id]*ut
             state_history && (x[i,:] = xt) 
             xt = psysd.A.M[:,:,ia]*xt + psysd.B.M[:,:,ib]*ut
         end
         return y, tout, x
      else
         @warn "Incommensurate time-step or final time less than one period not yet implemented"
         return nothing, nothing, nothing
      end
   end
end
"""
    psstepresp(sys[, tfinal]; ustep = ones(psys.nu), x0 = zeros(psys.nx), timesteps = 100, 
             state_history = false, abstol, reltol) -> (y, tout, x)

Compute the time response of a periodic system `sys = (A(t),B(t),C(t),D(t))` to step input signals. 
The final time `tfinal`, if not specified, is set equal to the period of the periodic system `sys`.
For a discrete-time system, the final time `tfinal` must be commensurate with the system sampling time `Ts`
(otherwise it is adjusted to the nearest smaller commensurate value). 
The keyword argument `ustep` is a vector with as many components 
as the inputs of `sys` and specifies the desired amplitudes of step inputs (default: all components are set to 1).   
The keyword argument `x0` is a vector, which specifies the initial state vector at time `0`, 
and is set to zero when omitted. 
The keyword argument `timesteps` specifies the number of desired simulation time steps 
(default: `timesteps = 100`). 

If `ns` is the total number of simulation values, `n` the number of state components, 
`p` the number of system outputs and `m` the number of system inputs, then
the resulting `ns×p×m` array `y` contains the resulting time histories of the outputs of `sys`, such 
that `y[:,:,j]` is the time response for the `j`-th input set to `ustep[j]` and the rest of inputs set to zero.  
The vector `tout` contains the corresponding values of the time samples.
The `i`-th row `y[i,:,j]` contains the output values at time `tout[i]` of the `j`-th step response.  
If the keyword parameter value `state_history = true` is used, then the resulting `ns×n×m` array`x` contains 
the resulting time histories of the state vector and 
the `i`-th row `x[i,:,j]` contains the state values at time `tout[i]` of the `j`-th step response.  
For a discrete-time periodic system with time-varying state dimensions, `n` is the maximum value of the 
dimensions of the state vector over one period. The components of `x[i,:,j]` have trailing zero values if the 
corresponding state vector dimension is less than `n`.  
By default, the state history is not saved and `x = nothing`.

The total number of simulation values `ns` is set as follows: for a continuous-time system `ns = timesteps+1` 
and for a discrete-time system `ns = min(timesteps,tfinal/Ts)+1`. 

For a continuous-time model an equivalent discretized model is determined to be used for simulation, provided
the time step `Δ = tfinal/timesteps` is commensurate with the system period `T` and `tfinal >= T`. 
The discretization is performed by determining the monodromy matrix as a product of 
state transition matrices of the extended state-space matrix `[A(t) B(t); 0 0]` 
by integrating numerically the corresponding homogeneous linear ODE.  
If the time step `Δ` is not commensurate with the period `T` or `tfinal < T`, then numerical integrations
of the underlying ODE systems are performed. 
The ODE solver to be employed can be 
specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
the fixed step length `dt` (default: `dt = Ts/10`) (see [`tvstm`](@ref)). 
For large numbers of product terms, parallel computation of factors can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable. 
"""
function psstepresp(psys::PeriodicStateSpace{PM}, tfinal::Real = 0;
                  x0::AbstractVector{<:Number} = zeros(T,psys.nx[1]), ustep::AbstractVector{<:Number} = ones(T,psys.nu[1]), 
                  state_history::Bool = false, timesteps::Int = 100,  
                  solver::String  = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where 
                  {T, PM <: Union{PeriodicArray{:d,T},PeriodicMatrix{:d,T},PeriodicFunctionMatrix{:c,T},HarmonicArray{:c,T},FourierFunctionMatrix{:c,T}}}

   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   n, p, m = psys.nx[1], psys.ny[1], psys.nu[1]

   m1 = length(ustep)
   m == m1 || error("ustep must have as many components as system inputs")
   n == length(x0) || error("x0 must have the same length as the system state vector")
 
   #disc = Domain == :d
   disc = isdiscrete(psys)
   ns = timesteps
   ns > 0 || error("Number of time steps must be positive")
   tfinal >= 0 || error("Final time must be positive")
   if disc
      Δ = psys.Ts 
      tf = tfinal == 0 ? psys.period : (rationalize(tfinal/Δ).den == 1 ? tfinal : (floor(tfinal/Δ)+1)*Δ)
      N = Int(round(tf/Δ))+1      # simulation points
      K = min(ns+1,N)             # number of values
      kstep = max(Int(floor((N-1)/ns)),1)
      #kstep += (kstep*ns != K) 
      Δs = Δ*kstep 
   else
      tf = tfinal == 0 ? psys.period : tfinal
      Δ = tf/ns
      N = ns+1
      Δs = Δ 
      K = N
   end
   tout = Vector{Float64}(0:Δs:(K-1)*Δs) 
   y = similar(Array{T1,3},K,p,m)
   xt = copy(x0)
   if disc
      pa = length(psys.A) 
      pb = length(psys.B)
      pc = length(psys.C) 
      pd = length(psys.D)
      #@show tfinal, tf, Δ, Δs, N, K, ns, kstep, min(timesteps,tfinal/Δ)+1
      if PM <: PeriodicArray
         state_history ?  x = similar(Array{T1,3},N,n,m) : x = nothing 
         for j = 1:m
             xt = copy(x0)
             ut = ustep[j]
             k = 0
             for i = 1:N 
                 ia = mod(i-1,pa)+1
                 ib = mod(i-1,pb)+1
                 ic = mod(i-1,pc)+1
                 id = mod(i-1,pd)+1
                 if mod(i-1,kstep) == 0 && k < K
                    k += 1
                    y[k,:,j] = psys.C.M[:,:,ic]*xt + psys.D.M[:,:,id][:,j]*ut
                    state_history && (x[k,:,j] = xt) 
                 end
                 xt = psys.A.M[:,:,ia]*xt + psys.B.M[:,:,ib][:,j]*ut
             end
         end
      else
         nmax = maximum(psys.nx)
         state_history ?  x = zeros(T1, K, nmax, m) : x = nothing 
         for j = 1:m
            xt = copy(x0)
            ut = ustep[j]
            k = 0
            for i = 1:N 
                ia = mod(i-1,pa)+1
                ib = mod(i-1,pb)+1
                ic = mod(i-1,pc)+1
                id = mod(i-1,pd)+1
                if mod(i-1,kstep) == 0 && k < K
                  k += 1
                  y[k,:,j] = psys.C.M[ic]*xt + psys.D.M[id][:,j]*ut
                  state_history && (copyto!(view(x,k,1:length(xt),j), xt)) 
                end
                xt = psys.A.M[ia]*xt + psys.B.M[ib][:,j]*ut
            end
         end
      end
      return y, tout, x
   else
      # for commensurate time-steps and final time larger than one period use discretized model
      if rationalize(psys.period/Δ).den == 1 && (tout[end] ≥ psys.period || tout[end] ≈ psys.period)
         state_history ?  x = similar(Array{T1,3},N,n,m) : x = nothing 
         dt == 0 && (dt = Δ/10)
         psysd = psc2d(psys, Δ; solver, reltol, abstol, dt)
         pa = length(psysd.A) 
         pb = length(psysd.B)
         pc = length(psysd.C) 
         pd = length(psysd.D)
         for j = 1:m
             xt = copy(x0)
             ut = ustep[j]
             for i = 1:N 
                 ia = mod(i-1,pa)+1
                 ib = mod(i-1,pb)+1
                 ic = mod(i-1,pc)+1
                 id = mod(i-1,pd)+1
               #   y[i,:,j] = psysd.C.M[ic]*xt + psysd.D.M[id][:,j]*ut
               #   state_history && (x[i,:,j] = xt) 
               #   xt = psysd.A.M[ia]*xt + psysd.B.M[ib][:,j]*ut
                 y[i,:,j] = psysd.C.M[:,:,ic]*xt + psysd.D.M[:,:,id][:,j]*ut
                 state_history && (x[i,:,j] = xt) 
                 xt = psysd.A.M[:,:,ia]*xt + psysd.B.M[:,:,ib][:,j]*ut
             end
         end
         return y, tout, x
      else
         state_history ?  x = similar(Array{T1,3},N,n,m) : x = nothing 
         xt = copy(x0)
         for j = 1:m
             xt = copy(x0)
             ut = ustep[j]
             uf = t-> ut
             Bp = psys.B[:,j]
             for i = 1:N
                 t = tout[i]
                 y[i,:,j] = tpmeval(psys.C,t)*xt + tpmeval(psys.D,t)[:,j]*ut
                 state_history && (x[i,:,j] = xt) 
                 @inbounds xt  = tvtimeresp(psys.A, Bp, i*Δ, (i-1)*Δ, uf, xt; solver, reltol, abstol, dt) 
             end
         end
         return y, tout, x
      end
   end
end
function psstepresp(psys::PeriodicStateSpace{PM}, tfinal::Real = 0; kwargs...) where {T, PM <: PeriodicSymbolicMatrix{:c,T}} 
   psstepresp(convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys), tfinal; kwargs...)
end
function psstepresp(psys::PeriodicStateSpace{PM}, tfinal::Real = 0; kwargs...) where {T, PM <: PeriodicTimeSeriesMatrix{:c,T}} 
   psstepresp(convert(PeriodicStateSpace{HarmonicArray},psys), tfinal; kwargs...)
end

"""
    pstimeresp(sys, u, t, x0 = zeros(sys.nx); state_history = false, solver, reltol, abstol) -> (y, tout, x)

Compute the time response of a continuous-time periodic system `sys = (A(t),B(t),C(t),D(t))` to the input signals 
described by `u` and `t`. The time vector `t` consists of regularly spaced time samples. 
The input `u` is specified as a vector of time dependent signals 
with as many components as the number of inputs of `sys`. 
The vector `x0` specifies the initial state vector at time `t[1]` and is set to zero when omitted. 

The matrix `y` contains the resulting time history of the outputs of `sys` and 
the vector `tout` contains the corresponding values of the time samples.
The `i`-th row of `y` contains the output values at time `tout[i]`.  
If the keyword parameter value `state_history = true` is used, then the matrix `x` contains 
the resulting time history of the state vector and its `i`-th row contains 
the state values at time `tout[i]`. By default, the state history is not computed and `x = nothing`.

The ODE solver to be employed can be 
specified using the keyword argument `solver` (see below), together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`), 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and 
the fixed step length `dt` (default: `dt = 0`), only used if `solver = "symplectic"`. 
Depending on the desired relative accuracy `reltol`, 
lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

"""
function pstimeresp(psys::PeriodicStateSpace{PM}, u::Function, t::AbstractVector{<:Real},  
                    x0::AbstractVector{<:Number} = zeros(T,psys.nx[1]);  
                    state_history::Bool = false, solver::String  = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where {T, PM <: AbstractPeriodicArray{:c,T}}
   T1 = T <: BlasFloat ? T : promote_type(Float64,T) 
   n = psys.nx 
   p, m = psys.ny, psys.nu
   m1 = length(u(0))
   m == m1 || error("u must have length as system inputs")
   n == length(x0) || error("x0 must have the same length as the system state vector")

   N = length(t)
   if N > 1
      Δ = t[2]-t[1]
      (any(diff(t) .<= 0) || any(isinf.(t)) || any(abs.(diff(t).-Δ).> 0.00001*Δ)) && 
           error("time vector t must contain monotonically increasing and evenly spaced time samples")
      tout = t
   else
      error("Time values must be given in the continuous-time case")
   end 
   p == 1 ? y = Vector{T1}(undef, N) : y = Matrix{T1}(undef, N, p) 
   x = state_history ? Matrix{T1}(undef, N, n) : nothing
   xt = copy(x0)
   for i = 1:N
       t = tout[i]
       ut = u(t)
       y[i,:] = tpmeval(psys.C,t)*xt + tpmeval(psys.D,t)*ut
       state_history && (x[i,:] = xt) 
       @inbounds xt  = tvtimeresp(psys.A, psys.B, i*Δ, (i-1)*Δ, u, xt; solver, reltol, abstol, dt) 
   end
   return y, tout, x
end
function tvtimeresp(A::PM, B::PM, tf, t0, u, x0::AbstractVector; solver = "", reltol = 1e-4, abstol = 1e-7, dt = 0) where
   {PM <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix,PeriodicTimeSeriesMatrix}} 
   """
      tvtimeresp(A, B, tf, t0, u, x0; solver, reltol, abstol) -> x::Vector

   Compute the solution at tf > t0 of the differential equation 
            .
            x(t) = A(t)*x(t)+B(t)*u(t), x(t0) = x0, tf > t0 . 

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt' (default: `dt = 0`, only used if `solver = "symplectic"`) (see [`tvstm`](@ref)). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   n == size(B,1) || error("the periodic matrix B must have same row dimension as A")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   tspan = (T(t0),T(tf))
   # ftresp(x,p,t) = tpmeval(A,t)*x + tpmeval(B,t)*u(t)
   # prob = ODEProblem(ftresp, x0, tspan)
   #ftresp!(dx,x,p,t) = mul!(dx,[tpmeval(A,t) tpmeval(B,t)],[x;u(t)])
   function ftresp!(dx,x,p,t)
      mul!(dx,tpmeval(A,t),x)
      mul!(dx,tpmeval(B,t),u(t),1,1)
   end
   
   prob = ODEProblem(ftresp!, x0, tspan)

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
   return sol[end]    
end
