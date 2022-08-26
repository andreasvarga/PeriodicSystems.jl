"""
    pcric(A, R, Q; K = 10, adj = false, solver, reltol, abstol, fast) -> (X, EVALS)

Solve the periodic Riccati differential equation

    .                                                 
    X(t) = A(t)X(t) + X(t)A(t)' + Q(t) - X(t)R(t)X(t) ,  if adj = false,

or 

     .                                                
    -X(t) = A(t)'X(t) + X(t)A(t) + Q(t) - X(t)R(t)X(t) , if adj = true. 

The periodic matrices `A`, `R` and `Q` must have the same type, the same dimensions and commensurate periods, 
and additionally `R` and `Q` must be symmetric. The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 

If `fast = true` (default) the multiple-shooting method is used in conjunction with fast pencil reduction techniques, as proposed in [1],
to determine the periodic solution in `t = 0` and a multiple point generator of the appropriate periodic differential Riccati equation
is determined  (see [2] for details). 
If `fast = false`, the multiple-shooting method is used in 
conjunction with the periodic Schur decomposition to determine multiple point generators directly from the stable periodic invariant subspace of 
an appropriate symplectic transition matrix (see also [2] for more details). 

The keyword argument `K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 10`). 
The multiple point generator is computed  by solving the appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 
The resulting periodic time-series matrix is finally converted to the appropriate periodic representation.

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) (see [`tvstm`](@ref)). 
For large values of `K`, parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.
    
"""
function pcric(A::PeriodicFunctionMatrix, R::PeriodicFunctionMatrix, Q::PeriodicFunctionMatrix; K::Int = 10, adj = false, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true)
   X, EVALS = pgcric(A, R, Q, K;  adj, solver, reltol, abstol, dt, fast)
   return convert(PeriodicFunctionMatrix,convert(HarmonicArray,X)), EVALS
end
function pcric(A::PeriodicSymbolicMatrix, R::PeriodicSymbolicMatrix, Q::PeriodicSymbolicMatrix; K::Int = 10, adj = false, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true)
   #X, EVALS = pgcric(convert(PeriodicFunctionMatrix,A), convert(PeriodicFunctionMatrix,R), convert(PeriodicFunctionMatrix,Q), K;  adj, solver, reltol, abstol, dt, fast)
   X, EVALS = pgcric(convert(HarmonicArray,A), convert(HarmonicArray,R), convert(HarmonicArray,Q), K;  adj, solver, reltol, abstol, dt, fast)
   return convert(PeriodicSymbolicMatrix,X), EVALS
end
function pcric(A::HarmonicArray, R::HarmonicArray, Q::HarmonicArray; K::Int = 10, adj = false, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true)
   X, EVALS = pgcric(A, R, Q, K;  adj, solver, reltol, abstol, dt, fast)
   return convert(HarmonicArray, X), EVALS
end
function pcric(A::FourierFunctionMatrix, R::FourierFunctionMatrix, Q::FourierFunctionMatrix; K::Int = 10, adj = false, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true)
   X, EVALS = pgcric(A, R, Q, K;  adj, solver, reltol, abstol, dt, fast)
   return convert(FourierFunctionMatrix, X), EVALS
end
function pcric(A::PeriodicTimeSeriesMatrix, R::PeriodicTimeSeriesMatrix, Q::PeriodicTimeSeriesMatrix; K::Int = 10, adj = false, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true)
   pgcric(convert(HarmonicArray,A), convert(HarmonicArray,R), convert(HarmonicArray,Q), K;  adj, solver, reltol, abstol, dt, fast)
end
for PM in (:PeriodicFunctionMatrix, :PeriodicSymbolicMatrix, :HarmonicArray, :FourierFunctionMatrix, :PeriodicTimeSeriesMatrix)
   @eval begin
      function prcric(A::$PM, B::$PM, R::$PM, Q::$PM; K::Int = 10, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true) 
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         tr = B*inv(R)*B'
         X, EVALS = pcric(A, (tr+tr')/2, Q; K, adj = true, solver, reltol, abstol, dt, fast)
         return X, EVALS, inv(R)*transpose(B)*X
      end
      function prcric(A::$PM, B::$PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true) 
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         tr = B*$PM(inv(R), A.period)*B'
         X, EVALS = pcric(A, (tr+tr')/2, $PM(Q, A.period); K, adj = true, solver, reltol, abstol, dt, fast)
         return X, EVALS, inv(R)*B'*X
      end
      function pfcric(A::$PM, C::$PM, R::$PM, Q::$PM; K::Int = 10, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true) 
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         tr = C'*inv(R)*C
         X, EVALS = pcric(A, (tr+tr')/2, Q; K, adj = false, solver, reltol, abstol, dt, fast)
         return X, EVALS, (X*C')*inv(R)
      end
      function pfcric(A::$PM, C::$PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, fast = true) 
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         tr = C'*$PM(inv(R), A.period; nperiod = A.nperiod)*C
         X, EVALS = pcric(A, (tr+tr')/2, $PM(Q, A.period; nperiod = A.nperiod); K, adj = false, solver, reltol, abstol, dt, fast)
         return X, EVALS, (X*C')*inv(R)
      end
   end
end
"""
    pfcric(A, C, R, Q; K = 10, solver, reltol, abstol, fast) -> (X, EVALS, F)

Compute the symmetric stabilizing solution `X(t)` of the periodic filtering related Riccati differential equation

    .                                                 -1 
    X(t) = A(t)X(t) + X(t)A(t)' + Q(t) - X(t)C(t)'R(t)  C(t)X(t) ,

the periodic stabilizing Kalman gain 

                         -1
    F(t) = -X(t)C(t)'R(t)   

and the corresponding stable characteristic multipliers `EVALS` of `A(t)-F(t)C(t)`. 

If `fast = true` (default) the multiple-shooting method is used in conjunction with fast pencil reduction techniques, while
if `fast = false`, the multiple-shooting method is used in 
conjunction with the periodic Schur decomposition approach. 

The periodic matrices `A`, `C`, `R` and `Q` must have the same type and commensurate periods, 
and additionally `R` must be symmetric positive definite and `Q` must be symmetric positive semidefinite. 
The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A`, `C`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 
"""
pfcric(A::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix, R::PeriodicFunctionMatrix, Q::PeriodicFunctionMatrix)
"""
    prcric(A, B, R, Q; K = 10, solver, reltol, abstol) -> (X, EVALS, F)

Compute the symmetric stabilizing solution `X(t)` of the periodic control related Riccati differential equation

     .                                                -1 
    -X(t) = A(t)'X(t) + X(t)A(t) + Q(t) - X(t)B(t)R(t)  B(t)'X(t) , 

the periodic stabilizing state-feedback gain 

                -1
    F(t) = -R(t)  B(t)'X(t) 

and the corresponding stable characteristic multipliers `EVALS` of `A(t)-B(t)F(t)`. 

If `fast = true` (default) the multiple-shooting method is used in conjunction with fast pencil reduction techniques, while
if `fast = false`, the multiple-shooting method is used in 
conjunction with the periodic Schur decomposition approach. 

The periodic matrices `A`, `B`, `R` and `Q` must have the same type and commensurate periods, 
and additionally `R` must be symmetric positive definite and `Q` must be symmetric positive semidefinite. 
The resulting symmetric periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B`, `R` and `Q` and the number of subperiods
is adjusted accordingly. 
"""
prcric(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix, R::PeriodicFunctionMatrix, Q::PeriodicFunctionMatrix)

"""
    pgcric(A, B, R, Q[, K = 1]; adj = false, solver, reltol, abstol) -> (X, EVALS)

Compute periodic generators for the periodic Riccati differential equation in the _filtering_ form

    .                                                  
    X(t) = A(t)X(t) + X(t)A(t)' + Q(t) - X(t)R(t)X(t), if adj = false,

or in the _control_ form

     .                                              
    -X(t) = A(t)'X(t) + X(t)A(t) + Q(t) - X(t)R(t)X(t) , if adj = true,

where `A(t)`, `R(t)` and `Q(t)` are periodic matrices of commensurate periods, 
with `A(t)` square, `R(t)` symmetric and positive definite,
and `Q(t)` symmetric and positive semidefinite. 
The resulting `X` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A(t)`, `R(t)` and `Q(t)`
are constant matrices and `N = K` otherwise. 
`EVALS` contains the stable characteristic multipliers of the monodromy matrix of 
the corresponding Hamiltonian matrix (also called closed-loop characteristic multipliers).
The period of `X` is set to the least common commensurate period of `A(t)`, `R(t)` and `Q(t)`
and the number of subperiods is adjusted accordingly. 
Any component matrix of `X` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method proposed in [1] is employed, first, to convert the continuous-time periodic Lyapunov into a discrete-time periodic Lyapunov equation satisfied by 
the generator solution in `K` time grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`) (see [`tvstm`](@ref)). 
For large values of `K`, parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function pgcric(A::PM1, R::PM3, Q::PM4, K::Int = 1; adj = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), 
   solver = "symplectic", reltol = 1e-4, abstol = 1e-7, dt = 0.0, fast = true) where
   {PM1 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix}, 
   PM3 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix},
   PM4 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix}} 

   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(R) || error("the periodic matrix R must have same dimensions as A")
   (n,n) == size(Q) || error("the periodic matrix Q must have same dimensions as A")

   period = promote_period(A, Q, R)
   na = Int(round(period/A.period))
   nq = Int(round(period/Q.period))
   nr = Int(round(period/R.period))
   nperiod = gcd(na*A.nperiod, nq*Q.nperiod, nr*R.nperiod)
   Ts = period/K/nperiod
   if isconstant(A) && isconstant(R) && isconstant(Q)
      if adj 
         X, EVALS, = arec(tpmeval(A,0),tpmeval(R,0), tpmeval(Q,0); rtol)
      else
         X, EVALS, = arec(tpmeval(A,0)', tpmeval(R,0), tpmeval(Q,0); rtol)
      end 
      return PeriodicTimeSeriesMatrix([X], period; nperiod), EVALS
   end

   T = promote_type(eltype(A),eltype(Q),eltype(R),Float64)
   n2 = n+n
   #hpd = Array{T,3}(undef, n2, n2, K) 
   i1 = 1:n; i2 = n+1:n2
   if K == 1
      hpd  = tvcric(A, R, Q, Ts, 0; adj, solver, reltol, abstol)
      SF = schur(hpd)
      select = abs.(SF.values) .< 1
      n == count(select .== true) || error("The symplectic pencil is not dichotomic")
      ordschur!(SF, select)
      x = SF.Z[i2,i1]/SF.Z[i1,i1]; x = (x+x')/2
      EVALS = SF.values[i1]
      X = PeriodicTimeSeriesMatrix([x], period; nperiod)
      ce = log.(complex(EVALS))/period
      return X, isreal(ce) ? real(ce) : ce 
   elseif !fast
      hpd = Array{T,3}(undef, n2, n2, K) 
      Threads.@threads for i = 1:K
         @inbounds hpd[:,:,i]  = tvcric(A, R, Q, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol) 
      end
      # this code is based on SLICOT tools
      S, Z, ev, sind, = PeriodicSystems.pschur(hpd)
      select = adj ? abs.(ev) .< 1 : abs.(ev) .> 1
      psordschur!(S, Z, select; sind)
      EVALS =  adj ? ev[select] : ev[.!select]
      X = similar(Vector{Matrix{T}},K)
      for i = 1:K
          #x = Z1[i][i2,i1]/Z1[i][i1,i1];  
          x = Z[i2,i1,i]/Z[i1,i1,i];  
          x = (x+x')/2
          X[i] = x
      end
      
      # this experimental code is based on tools provided in the PeriodicSchurDecompositions package
      # hpd = Vector{Matrix{T}}(undef, K) 
      # Threads.@threads for i = 1:K
      #    @inbounds hpd[i]  = tvcric(A, R, Q, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol, dt) 
      # end
      # PSF = PeriodicSchurDecompositions.pschur(hpd,:L)
      # select = adj ? abs.(PSF.values) .< 1 : abs.(PSF.values) .> 1
      # @show PSF.values
      # ordschur!(PSF, select)
      # EVALS = adj ? PSF.values[i1] : PSF.values[i2]
      # X = similar(Vector{Matrix{T}},K)
      # for i = 1:K
      #     x = PSF.Z[i][i2,i1]/PSF.Z[i][i1,i1];  
      #     x = (x+x')/2
      #     X[i] = x
      # end
      ce = log.(complex(EVALS))/period
      return PeriodicTimeSeriesMatrix(X, period; nperiod), isreal(ce) ? real(ce) : ce    
   else
      hpd = Vector{Matrix{T}}(undef, K) 
      Threads.@threads for i = 1:K
         @inbounds hpd[i]  = tvcric(A, R, Q, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol, dt) 
      end
      a, e = psreduc_reg(hpd)
      # Compute the ordered QZ decomposition with large eigenvalues in the
      # leading positions. Only Z is used.
      # Note: eigenvalues of this pencil have a tendency
      # to deflate out in the ``desired'' order (large values first)
      SF = schur!(e, a)  # use (e,a) instead (a,e) 
      # this code may produce inaccurate small characteristic multipliers for large values of K
      # the following test try to detect the presence of infinite values
      all(isfinite.(SF.values)) || @warn "possible accuracy loss"
      #@show a, e, SF.values
      select = adj ? abs.(SF.values) .> 1 : abs.(SF.values) .< 1
      n == count(select .== true) || error("The simplectic pencil is not dichotomic")
      ordschur!(SF, select)
      EVALS = adj ? SF.values[i2] : SF.values[i1]
      # compute the periodic generator in t = 0
      x = SF.Z[i2,i1]/SF.Z[i1,i1]; x = (x+x')/2
      X = similar(Vector{Matrix{T}},K)
      xn = x; xlast = zeros(eltype(x),n,n); kit = 0; 
      tol = 10*eps()*norm(xn,1)
      #@show norm(xn-xlast,1) 
      while norm(xn-xlast,1) > tol && kit <= 3
       kit += 1
       if adj
         for i in K:-1:1
             x = (x*hpd[i][i1,i2]-hpd[i][i2,i2])\(hpd[i][i2,i1]-x*hpd[i][i1,i1]) 
             x = (x+x')/2;
             X[i] = x;
         end
       else
         for i in 1:K
             x = (hpd[i][i2,i1]+hpd[i][i2,i2]*x)/(hpd[i][i1,i1]+hpd[i][i1,i2]*x)
             x = (x+x')/2
             i < K ? X[i+1] = x : X[1] = x
         end
       end
       xlast = xn 
       xn = X[1]
       #@show norm(xn-xlast,1) 
      end

      ce = log.(complex(EVALS))/period
      return PeriodicTimeSeriesMatrix(X, period; nperiod), isreal(ce) ? real(ce) : ce  
   end
end
function tvcric(A::PM1, R::PM3, Q::PM4, tf, t0; adj = false, solver = "symplectic", reltol = 1e-4, abstol = 1e-7, dt = 0.0) where
    {PM1 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix}, 
    PM3 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix},
    PM4 <: Union{PeriodicFunctionMatrix,HarmonicArray,FourierFunctionMatrix}} 
    """
       tvcric(A, R, Q, tf, t0; adj, solver, reltol, abstol, dt) -> Φ::Matrix

    Compute the state transition matrix for a linear Hamiltonian ODE with periodic time-varying coefficients. 
    For the given periodic matrices `A(t)`, `R(t)`, `Q(t)`, with `A(t)` square, `R(t)` and `Q(t)` symmetric, 
    and the initial time `t0` and final time `tf`, the state transition matrix `Φ(tf,t0)`
    is computed by integrating numerically the homogeneous linear ODE 
         
               dΦ(t,t0)/dt = H(t)Φ(t,t0),  Φ(t0,t0) = I
         
    on the time interval `[t0,tf]`. `H(t)` is a periodic Hamiltonian matrix defined as
    
                                  
         H(t) = [ -A'(t)  R(t) ],   if adj = false
                [  Q(t)   A(t) ]
                      
 
    or 
                                 
         H(t) = [  A(t)  -R(t)  ],  if adj = true. 
                [ -Q(t)  -A'(t) ]

    The ODE solver to be employed can be specified using the keyword argument `solver` (see [`tvstm`](@ref)), 
    (default: `solver = "symplectic"`) together with
    the required relative accuracy `reltol` (default: `reltol = 1.e-4`), 
    absolute accuracy `abstol` (default: `abstol = 1.e-7`) and/or 
    the fixed step length `dt` (default: `dt =  min(A.period/A.nperiod/100,tf-t0)`). 
    Depending on the desired relative accuracy `reltol`, 
    lower order solvers are employed for `reltol >= 1.e-4`, 
    which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
    higher order solvers are employed able to cope with high accuracy demands. 
    """
    n = size(A,1)
    T = promote_type(typeof(t0), typeof(tf))
    period = promote_period(A, R, Q)

    # using OrdinaryDiffEq
    n2 = n+n
    u0 = Matrix{T}(I,n2,n2)
    tspan = (T(t0),T(tf))
    H = adj ? PeriodicFunctionMatrix(t -> [ tpmeval(A,t) -tpmeval(R,t); -tpmeval(Q,t) -tpmeval(A,t)'], period) :
              PeriodicFunctionMatrix(t -> [-tpmeval(A,t)' tpmeval(R,t); tpmeval(Q,t) tpmeval(A,t)], period) 

    if solver != "linear" 
       #f!(du,u,p,t) = mul!(du,A.f(t),u)
       #fcric!(du,u,p,t) = mul!(du,tpmeval(H,t),u)
       #prob = ODEProblem(fcric!, u0, tspan)
       prob = ODEProblem(HamODE!, u0, tspan, (adj,A,R,Q) )
    end

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
    elseif solver == "linear" 
       iszero(dt) && (dt = min(A.period/A.nperiod/100,tf-t0))
       function update_func!(A,u,p,t)
            A .= p(t)
       end
       DEop = DiffEqArrayOperator(ones(T,n2,n2),update_func=update_func!)     
         #prob = ODEProblem(DEop, u0, tspan, A.f)
       prob = ODEProblem(DEop, u0, tspan, t-> tpmeval(H,t))
       sol = solve(prob,MagnusGL6(), dt = dt, save_everystep = false)
    elseif solver == "symplectic" 
       # high accuracy symplectic
       adaptive = iszero(dt) ? true : false
       sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive, dt, reltol, abstol, save_everystep = false)
    else 
       if reltol > 1.e-4  
          # low accuracy automatic selection
          sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
       else
          # high accuracy automatic selection
          sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
       end
    end
    return sol(tf)  
end
function HamODE!(du, u, pars, t)
   (adj, A, R, Q) = pars
   adj ? mul!(du,[ tpmeval(A,t) -tpmeval(R,t); -tpmeval(Q,t) -tpmeval(A,t)'],u) :
         mul!(du,[-tpmeval(A,t)' tpmeval(R,t); tpmeval(Q,t) tpmeval(A,t)],u) 
end
