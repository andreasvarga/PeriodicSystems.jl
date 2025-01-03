"""
     pdlqr(psys, Q, R, S; kwargs...) -> (F, EVALS)

Compute the optimal periodic stabilizing gain matrix `F(t)`, such that for a discrete-time periodic state-space model 
`psys` of the form
   
    x(t+1) = A(t)x(t) + B(t)u(t) + Bw(t)w(t) 
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t)

the state feedback control law

     u(t) = F(t)x(t) 
     
minimizes the quadratic index

     J = Sum {x'(t)Q(t)x(t) + u'(t)R(t)u(t) + 2*x'(t)S(t)u(t)}  

For a system of order `n(t)` with `m` control inputs in `u(t)`, `Q(t)` and `R(t)` are `n(t)×n(t)` and `m×m` symmetric matrices, respectively, and `S(t)` is an `n(t)×m` matrix.                
The matrix `S` is set to zero when omitted. The dimension of `u(t)` is deduced from the dimension of `R(t)`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)+B(t)F(t)`.

The keyword arguments contained in `kwargs` are those used for the function 
[`PeriodicMatrixEquations.prdric`](@extref).

Note: The pair `(A(t),B(t))` must be stabilizable and `[Q S;S' R]` must be nonnegative definite.
"""
function pdlqr(psys::PeriodicStateSpace{PM}, Q, R, S = missing; kwargs...) where {PM <:Union{PeriodicMatrix,PeriodicArray}}
   mu = size(R,1)[1]
   _, EVALS, F = prdric(psys.A, psys.B[:,1:mu], R, Q, S; kwargs...)
   EVALS = complex(EVALS).^(1/F.dperiod)
   return -F, EVALS
end
"""
     pclqr(psys, Q, R, S; intpol = true, kwargs...) -> (F, EVALS)

Compute the optimal periodic stabilizing gain matrix `F(t)`, such that for a continuous-time periodic state-space model 
`psys` of the form
   
      .
      x(t) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t),
 
the state feedback control law

     u(t) = F(t)x(t) 
     
minimizes the quadratic index
  
     J = Integral {x'(t)Q(t)x(t) + u(t)'R(t)u(t) + 2*x'(t)S(t)u(t)} dt.     

For a system of order `n` with `m` inputs, `Q(t)` and `R(t)` are `n×n` and `m×m` symmetric matrices, respectively, and `S(t)` is an `n×m` matrix.                
The matrix `S` is set to zero when omitted. The dimension of `u(t)` is deduced from the dimension of `R(t)`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)+B(t)F(t)`.

For `intpol = true` (default), the resulting periodic gain `F(t)` is computed from the 
stabilizing solution of a continuous-time periodic matrix differential Riccati equation using interpolation based formulas. 
If `intpol = false`, the gain `F(t)` is computed from a multi-point solution of the Riccati differential equation 
by the integration of the corresponding ODE using the nearest point value as initial condition. 
This option is not recommended to be used jointly with symplectic integrators, which are used by default.   

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.prcric`](@extref) (excepting intpol). 

Note: The pair `(A(t),B(t))` must be stabilizable, `R` must be positive definite and `[Q S;S' R]` must be nonnegative definite .
"""
function pclqr(psys::PeriodicStateSpace{PM}, Q, R, S = missing; intpol = true, kwargs...) where {PM <: Union{PeriodicFunctionMatrix,HarmonicArray}}
   period = psys.period
   if isa(R,AbstractMatrix) 
      minimum(real(eigvals(R))) > 0 || throw(ArgumentError("R must be positive definite")) 
      Rt = PM(R,period)
   else
      minimum(real(eigvals(R(rand())))) > 0 || throw(ArgumentError("R must be positive definite")) 
      Rt = R
   end
   mu = size(R,1); m = size(psys.B,2)
   mu > m && throw(ArgumentError("R must have order at most $m"))
   if isa(Q,AbstractMatrix) 
      minimum(real(eigvals(Q))) >= 0 || throw(ArgumentError("Q must be non-negative definite")) 
      Qt = PM(Q,period)
   else
      minimum(real(eigvals(Q(rand())))) >= 0 || throw(ArgumentError("Q must be non-negative definite")) 
      Qt = Q
   end
   if ismissing(S) || iszero(S) 
      _, EVALS, F = prcric(psys.A, psys.B[:,1:mu], Rt, Qt; intpol, kwargs...) 
   else
      St = isa(S,AbstractMatrix) ? PM(S,period) : S
      RiS = inv(Rt)*St'; Qu = pmmuladdsym(Qt, St, RiS,(1,-1))
      minimum(real(eigvals(Qu(rand())))) >= 0 || throw(ArgumentError("Q-S*inv(R)*S' must be non-negative definite")) 
      _, EVALS, F = prcric(psys.A-psys.B[:,1:mu]*RiS, psys.B[:,1:mu], Rt, Qu; intpol = true, kwargs...)  
      F = F+RiS
   end
   return -F, EVALS
end
"""
     pdlqry(psys, Q, R, S; kwargs...) -> (F, EVALS)

Compute the optimal periodic stabilizing gain matrix `F(t)`, such that for a periodic discrete-time 
state-space model `psys` of the form
   
    x(t+1) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t),

the state feedback control law

     u(t) = F(t)x(t) 
     
minimizes the quadratic index
  
     J = Sum {y'(t)Q(t)y(t) + u'(t)R(t)u(t) + 2*y'(t)S(t)u(t)} 

For a system with `m` control inputs `u(t)` and `p` outputs, `Q(t)` and `R(t)` are `p×p` and `m×m` symmetric matrices, respectively, and `S(t)` is a `p×m` matrix.  
The dimension of `u(t)` is deduced from the dimension of `R(t)`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices.               
The matrix `S` is set to zero when omitted. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)+B(t)F(t)`.

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.prdric`](@extref). 

Note: The pair `(A(t),B(t))` must be stabilizable and `[Q S;S' R]` must be nonnegative definite.
"""
function pdlqry(psys::PeriodicStateSpace{PM}, Q, R, S = missing; kwargs...) where {PM <:Union{PeriodicMatrix,PeriodicArray}}
   # Explicitly form [QQ SS; SS' RR] := [C D;0 I]'*[Q S;S' R]*[C D;0 I]
   mu = size(R,1)
   QQ = pmtrmulsym(psys.C, Q*psys.C)
   Du = psys.D[:,1:mu]
   QD = Q*Du

   if ismissing(S)
      RR = pmmultraddsym(R, Du, QD)
      SS = psys.C'*QD
   else
      Y = S'*Du; pmsymadd!(Y); 
      RR = pmmultraddsym(R+Y, Du, QD)
      SS = psys.C'*(QD+S)
   end
   
   _, EVALS, F = prdric(psys.A, psys.B[:,1:mu], RR, QQ, SS; kwargs...)
   EVALS = EVALS.^(1/F.dperiod)
   return -F, EVALS
end
"""
     pclqry(psys, Q, R, S; intpol = true, kwargs...) -> (F, EVALS)

Compute the optimal periodic stabilizing gain matrix F(t), such that for a periodic 
continuous-time state-space model `psys` of the form 
      .
      x(t) = A(t)x(t) + B(t)u(t) + Bw(t)w(t) 
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) ,
 
the state feedback control law

     u(t) = F(t)x(t) 
     
minimizes the quadratic index
  
     J = Integral {y'(t)Q(t)y(t) + u'(t)R(t)u(t) + 2*y'(t)S(t)u(t)} dt 

For a system with `m` control inputs `u(t)` and `p` outputs, `Q(t)` and `R(t)` are `p×p` and `m×m` symmetric matrices, respectively, and `S(t)` is an `p×m` matrix. 
The dimension of `u(t)` is deduced from the dimension of `R(t)`. 
`R`, `Q` and `S` can be alternatively provided as constant real matrices.                             
The matrix `S` is set to zero when omitted. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)+B(t)F(t)`.

For `intpol = true` (default), the resulting periodic gain `F(t)` is computed from the 
stabilizing solution of a continuous-time periodic matrix differential Riccati equation using interpolation based formulas. 
If `intpol = false`, the gain `F(t)` is computed from a multi-point solution of the Riccati differential equation 
by the integration of the corresponding ODE using the nearest point value as initial condition. 
This option is not recommended to be used jointly with symplectic integrators, which are used by default.   

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.prcric`](@extref) (excepting intpol).  

Note: The pair `(A(t),B(t))` must be stabilizable and `[Q S;S' R]` must be nonnegative definite.
"""
function pclqry(psys::PeriodicStateSpace{<:PM}, Q, R, S = missing; intpol = true, kwargs...) where {PM <: Union{PeriodicFunctionMatrix,HarmonicArray}}
   # Explicitly form [QQ SS;SS' RR] = [C D;0 I]'*[Q S;S' R]*[C D;0 I]

   mu = size(R,1)
   QQ = pmtrmulsym(psys.C, Q*psys.C)
   Du = psys.D[:,1:mu]
   QD = Q*Du

   if ismissing(S)
      RR = pmmultraddsym(R, Du, QD)
      SS = psys.C'*QD
   else
      Y = S'*Du; pmsymadd!(Y); 
      RR = pmmultraddsym(R+Y, Du, QD)
      SS = psys.C'*(QD+S)
   end
   
   if iszero(SS) 
      _, EVALS, F = prcric(psys.A, psys.B[:,1:mu], RR, QQ; intpol, kwargs...) 
   else
      # Qu = QQ-SS*RiS; 
      RiS = inv(RR)*SS'; Qu = pmmuladdsym(QQ, SS, RiS, -1, 1)
      _, EVALS, F = prcric(psys.A-psys.B[:,1:mu]*RiS, psys.B[:,1:mu], RR, Qu; intpol, kwargs...)  
      F = F+RiS
   end
   return -F, EVALS
end
"""
     pdkeg(psys, Qw, Rv, Sn; kwargs...) -> (L, EVALS)

Compute the Kalman estimator periodic gain matrix `L(t)` for a discrete-time periodic state-space model 
`psys` of the form
   
    x(t+1) = A(t)x(t) + B(t)u(t) + w(t)
      y(t) = C(t)x(t) + D(t)u(t) + v(t)
 
and the noise covariance data `E{w(t)w(t)'} = Qw(t)`, `E{v(t)v(t)'} = Rv(t)`, `E{w(t)v(t)'} = Sn(t)`, 
for a Kalman estimator 

      xe(t+1) = A(t)xe(t) + B(t)u(t) + L(t)(y(t)-C(t)xe(t)-D(t)u(t))

For a system of order `n` with `p` outputs, `Qw` and `Rv` are `n×n` and `p×p` symmetric matrices, respectively, and `Sn` is an `n×p` matrix.                
`Qw`, `Rv` and `Sn` can be alternatively provided as constant real matrices.               
The matrix `Sn` is set to zero when omitted. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)-L(t)C(t)`.

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.pfdric`](@extref). 

Note: The pair `(A(t),C(t))` must be detectable and `[Qw Sn;Sn' Rv]` must be nonnegative definite.
"""
function pdkeg(psys::PeriodicStateSpace{PM}, Qw, Rv, Sn = missing; kwargs...) where {PM <:Union{PeriodicMatrix,PeriodicArray}}
   _, EVALS, L = pfdric(psys.A, psys.C, Rv, Qw, Sn; kwargs...)
   EVALS = complex(EVALS).^(1/L.dperiod)
   return L, EVALS
end
"""
     pckeg(psys, Qw, Rv, Sn; intpol = true, kwargs...) -> (L, EVALS)

Compute the Kalman estimator periodic gain matrix `L(t)` for a continuous-time periodic state-space model 
`psys` of the form
   
      .
      x(t) = A(t)x(t) + B(t)u(t) + w(t)
      y(t) = C(t)x(t) + D(t)u(t) + v(t)
 
and the noise covariance data `E{w(t)w(t)'} = Qw(t)`, `E{v(t)v(t)'} = Rv(t)`, `E{w(t)v(t)'} = Sn(t)`, 
for a Kalman estimator 

       .
      xe(t) = A(t)xe(t) + B(t)u(t) + L(t)(y(t)-C(t)xe(t)-D(t)u(t))

For a system of order `n` with `p` outputs, `Qw` and `Rv` are `n×n` and `p×p` symmetric matrices, respectively, and `Sn` is an `n×p` matrix.                
`Qw`, `Rv` and `Sn` can be alternatively provided as constant real matrices. 
The matrix `Sn` is set to zero when omitted. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)-L(t)C(t)`.

For `intpol = true` (default), the resulting periodic gain `L(t)` is computed from the 
stabilizing solution of a continuous-time periodic matrix differential Riccati equation using interpolation based formulas. 
If `intpol = false`, the gain `L(t)` is computed from a multi-point solution of the Riccati differential equation 
by the integration of the corresponding ODE using the nearest point value as initial condition. 
This option is not recommended to be used jointly with symplectic integrators, which are used by default.   

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.pfcric`](@extref) (excepting intpol). 

Note: The pair `(A(t),C(t))` must be detectable and `[Qw Sn;Sn' Rv]` must be nonnegative definite.
"""
function pckeg(psys::PeriodicStateSpace{PM}, Qw, Rv, Sn = missing; intpol = true, kwargs...) where  {PM <: Union{PeriodicFunctionMatrix,HarmonicArray}}
   period = psys.period
   if isa(Rv,AbstractMatrix) 
      minimum(real(eigvals(Rv))) > 0 || throw(ArgumentError("Rv must be positive definite")) 
      Rt = PM(Rv,period)
   else
      minimum(real(eigvals(Rv(rand())))) > 0 || throw(ArgumentError("Rv must be positive definite")) 
      Rt = Rv
   end
   if isa(Qw,AbstractMatrix) 
      minimum(real(eigvals(Qw))) >= 0 || throw(ArgumentError("Qw must be non-negative definite")) 
      Qt = PM(Qw,period)
   else
      minimum(real(eigvals(Qw(rand())))) >= 0 || throw(ArgumentError("Qw must be non-negative definite")) 
      Qt = Qw
   end
   if ismissing(Sn) || iszero(Sn) 
      _, EVALS, L = pfcric(psys.A, psys.C, Rt, Qt; intpol, kwargs...) 
   else
      St = isa(Sn,AbstractMatrix) ? PM(Sn,period) : Sn
      RiS = St*inv(Rt); Qwv = pmmuladdtrsym(Qt, RiS, St, (1,-1))
      minimum(real(eigvals(Qwv(rand())))) >= 0 || throw(ArgumentError("Qw-Sn*inv(Rv)*Sn' must be non-negative definite")) 
      _, EVALS, Lw = pfcric(psys.A-RiS*psys.C, psys.C, Rt, Qwv; intpol = true, kwargs...)  
      L = Lw+RiS
   end
   return L, EVALS
end
"""
     pdkegw(psys, Qw, Rv, Sn; kwargs...) -> (L, EVALS)

Compute the Kalman estimator periodic gain matrix `L(t)` for a discrete-time periodic state-space model 
`psys` of the form
   
    x(t+1) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) + v(t)
 
and the noise covariance data `E{w(t)w(t)'} = Qw(t)`, `E{v(t)v(t)'} = Rv(t)`, `E{w(t)v(t)'} = Sn(t)`, 
for a Kalman estimator 

      xe(t+1) = A(t)xe(t) + B(t)u(t) + L(t)(y(t)-C(t)xe(t)-D(t)u(t))

For a system with `mw` disturbance inputs and `p` outputs, `Qw` and `Rv` are `mw×mw` and `p×p` symmetric matrices, respectively, and `Sn` is an `mw×p` matrix.  
`Qw`, `Rv` and `Sn` can be alternatively provided as constant real matrices.               
The matrix `Sn` is set to zero when omitted. The number of disturbance inputs `mw` is defined by the order of matrix `Qw`. 

Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)-L(t)C(t)`.

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.pfdric`](@extref). 

Note: The pair `(A(t),C(t))` must be detectable,  `Qw` must be non-negative definite,  `Rv` must be positive definite
and `[Qw Sn; Sn' Rv]` must be nonnegative definite.
"""
function pdkegw(psys::PeriodicStateSpace{PM}, Qw, Rv, Sn = missing; kwargs...) where {PM <:Union{PeriodicMatrix,PeriodicArray}}
   mw = size(Qw,1)
   p, m = size(psys)
   Bw = psys.B[:,m-mw+1:m]; Dw = psys.D[:,m-mw+1:m]; 
   # Explicitly form aggregate covariance matrices
   #
   #  [ Qb  Sb ]     [ Bw  0 ] [ Qw  Sn ] [ Bw'  Dw' ]
   #  [        ]  =  [       ] [        ] [          ]
   #  [ Sb' Rb ]     [ Dw  I ] [ Sn  Rv ] [  0    I  ]
   #

   Qb = pmmultrsym(Bw*Qw, Bw)
   Qt = Dw*Qw
   if ismissing(Sn)
      Rb = pmmuladdtrsym(Rv, Qt, Dw)
      Sb = Bw*Qt'
   else
      Y = Dw*Sn; pmsymadd!(Y); 
      Rb = pmmuladdtrsym(Rv+Y, Qt, Dw)
      Sb = Bw*(Qt'+Sn)
   end

   _, EVALS, L = pfdric(psys.A, psys.C, Rb, Qb, Sb; kwargs...)
   EVALS = complex(EVALS).^(1/L.dperiod)
   return L, EVALS
end
"""
     pckegw(psys, Qw, Rv, Sn; intpol = true, kwargs...) -> (L, EVALS)

Compute the Kalman estimator periodic gain matrix `L(t)` for a continuous-time periodic state-space model 
`psys` of the form
   
      .
      x(t) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) + v(t)
 
and the noise covariance data `E{w(t)w(t)'} = Qw(t)`, `E{v(t)v(t)'} = Rv(t)`, `E{w(t)v(t)'} = Sn(t)`, 
for a Kalman estimator 

       .
      xe(t) = A(t)xe(t) + B(t)u(t) + L(t)(y(t)-C(t)xe(t)-D(t)u(t))

For a system with `mw` disturbance inputs and `p` outputs, `Qw` and `Rv` are `mw×mw` and `p×p` symmetric matrices, respectively, and `Sn` is an `mw×p` matrix.                
`Qw`, `Rv` and `Sn` can be alternatively provided as constant real matrices.               
The matrix `Sn` is set to zero when omitted. The number of disturbance inputs `mw` is defined by the order of matrix `Qw`. 
Also returned are the closed-loop characteristic exponents `EVALS` of `A(t)-L(t)C(t)`.

For `intpol = true` (default), the resulting periodic gain `L(t)` is computed from the 
stabilizing solution of a continuous-time periodic matrix differential Riccati equation using interpolation based formulas. 
If `intpol = false`, the gain `L(t)` is computed from a multi-point solution of the Riccati differential equation 
by the integration of the corresponding ODE using the nearest point value as initial condition. 
This option is not recommended to be used jointly with symplectic integrators, which are used by default.   

The keyword arguments contained in `kwargs` are those used for the function [`PeriodicMatrixEquations.pfcric`](@extref) (excepting intpol). 

Note: The pair `(A(t),C(t))` must be detectable,  `Qw` must be non-negative definite,  `Rv` must be positive definite
and `[Qw Sn; Sn' Rv]` must be nonnegative definite.
"""
function pckegw(psys::PeriodicStateSpace{PM}, Qw, Rv, Sn = missing; intpol = true, kwargs...) where  {PM <: Union{PeriodicFunctionMatrix,HarmonicArray}}
   mw = size(Qw,1)
   p, m = size(psys)
   Bw = psys.B[:,m-mw+1:m]; Dw = psys.D[:,m-mw+1:m]; 
   # period = psys.period
   # if isa(Rv,AbstractMatrix) 
   #    minimum(real(eigvals(Rv))) > 0 || throw(ArgumentError("Rv must be positive definite")) 
   #    Rt = PM(R,period)
   # else
   #    minimum(real(eigvals(Rv(rand())))) > 0 || throw(ArgumentError("Rv must be positive definite")) 
   #    Rt = R
   # end
   # if isa(Qw,AbstractMatrix) 
   #    minimum(real(eigvals(Qw))) >= 0 || throw(ArgumentError("Qw must be non-negative definite")) 
   #    Qt = PM(Qw,period)
   # else
   #    minimum(real(eigvals(Qw(rand())))) >= 0 || throw(ArgumentError("Qw must be non-negative definite")) 
   #    Qt = Qw
   # end
   # Explicitly form aggregate covariance matrices
   #
   #  [ Qb  Sb ]     [ Bw  0 ] [ Qw  Sn ] [ Bw'  Dw' ]
   #  [        ]  =  [       ] [        ] [          ]
   #  [ Sb' Rb ]     [ Dw  I ] [ Sn  Rv ] [  0    I  ]
   #
   Qb = pmmultrsym(Bw*Qw, Bw)
   Qt = Dw*Qw
   if ismissing(Sn) || iszero(Sn) 
      Rb = pmmuladdtrsym(Rv, Qt, Dw)
      Sb = Bw*Qt'
      if iszero(Sb)
         _, EVALS, L = pfcric(psys.A, psys.C, Rb, Qb; intpol, kwargs...) 
      else
         RiS = Sb*inv(Rb); Qu = pmmuladdtrsym(Qb, RiS, Sb, (1,-1))
         _, EVALS, Lw = pfcric(psys.A-RiS*psys.C, psys.C, Rb, Qu; intpol, kwargs...)  
         L = Lw+RiS
      end
   else
      Y = Dw*Sn; pmsymadd!(Y); 
      Rb = pmmuladdtrsym(Rv+Y, Qt, Dw)
      Sb = Bw*(Qt'+Sn)
      RiS = Sb*inv(Rb); Qu = pmmuladdtrsym(Qb, RiS, Sb, (1,-1))
      if isa(Qu,AbstractMatrix) 
         minimum(real(eigvals(Qu))) >= 0 || throw(ArgumentError("Qw-Sn*inv(Rv)*Sn' must be non-negative definite")) 
      else
         minimum(real(eigvals(Qu(rand())))) >= 0 || throw(ArgumentError("Qw-Sn*inv(Rv)*Sn' must be non-negative definite")) 
      end
      _, EVALS, Lw = pfcric(psys.A-RiS*psys.C, psys.C, Rb, Qu; intpol, kwargs...)  
      L = Lw+RiS
   end
   return L, EVALS
end
"""
    pdlqofc(psys, Q, R; S, G = I, sdeg = 1, stabilizer, optimizer, vinit, maxiter, vtol, Jtol, gtol, show_trace) -> (Fopt, info)

Compute for the discrete-time periodic state-space system `psys = (A(t),B(t),C(t),D(t))` of the form
   
    x(t+1) = A(t)x(t) + B(t)u(t) + Bw(t)w(t) 
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) ,

the optimal periodic feedback gain `Fopt(t)` in the output feedback control law  

    u(t) = Fopt(t)*y(t), 
    
which minimizes the expectation of the quadratic index

     J = E{ Sum [x(t)'*Q(t)*x(t) + 2*x(t)'*S(t)*u(t) + u(t)'*R(t)*u(t)] },

where `Q(t)`, `R(t)` and `S(t)` are periodic weighting matrices. 
For a system of order `n` with `m` control inputs in `u(t)` and `p` measurable outputs in `y(t)`, 
`Q(t)` and `R(t)` are `n×n` and `m×m` symmetric periodic matrices, respectively, and 
`S(t)` is an `n×m` periodic matrix, which can be specified via the keyword argument `S`. 
By default `S = missing`, in which case, `S(t) = 0` is assumed.   
The periodic matrices `Q(t)`,`R(t)` and `S(t)` have the same type as the matrices of the state-space system,
i.e., either of type `PeriodicMatrix` or `PeriodicArray`.  
The dimension `m` of `u(t)` is deduced from the dimension of `R(t)`. 
`Q`, `R` and `S` can be alternatively provided as constant real matrices. 

The resulting `m×p` periodic output feedback gain `Fopt(t)` has the same type as the state-space system matrices
and is computed as `Fopt(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` 
defined as 

     F(t) = F_i  for i ∈ {1, ..., ns} 
           
where `ns` is the number of sampling times in a period (i.e., `ns = psys.period/psys.Ts`) and `F_i` is the `i`-th gain. 

The covariance of the initial state `x(0)` can be specified via the keyword argument `G` (default: `G = I`)
and a desired stability degree of the closed-loop characteristic multipliers can be specified using
the keyword argument `sdeg` (default: `sdeg = 1`). 

For the determination of the optimal feedback gains `F_i` for `i = 1, ...., ns` an optimization-based approach is employed using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-based limited-memory quasi-Newton method (also known as `L-BFGS`) for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`, where 
an initial step length for the line search algorithm is chosen using the keyword argument `alphaguess` 
(see the [`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl) package for alternative options). 
The employed default line search algorithm is `HagerZhang()` and an alternative method can be specified using the keyword argument `linesearch` 
(e.g., `linesearch = LineSearches.MoreThuente()`).  
Alternative gradient-based methods can be also selected, such as, for example,  the quasi-Newton method `BFGS` with `optimizer = Optim.BFGS()`, or 
for small size optimization problems, the Nelder-Mead gradient-free method with `optimizer = Optim.NelderMead()`. 
For the computation of the function `J` and its gradient  `∇J`, the formulas developed in [1] for stable systems are used. Each evaluation involves the solution of 
of a pair of periodic Lyapunov difference equations using the method proposed in [2].  
If the original system `psys` is unstable, the computation of a stabilizing feedback is performed using the same optimization techniques applied iteratively to systems 
with modified state matrices of the form  `A(t)/α` and control matrices `B(t)/α`,
where `α ≥ 1` is chosen such that `A(t)/α` is stable, and the values of `α` are successively decreased until the stabilization is achieved with `α = 1`.
The optimization method for stabilization can be independently selected using the keyword argument `stabilizer`, with the default setting  
`stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`. If only stabilization is desired, then use  `optimizer = nothing`. 

An internal optimization variable `v` is used, formed as an `m×p×ns` array with `v[:,:,i] := F_i`, for `i = 1, ..., ns`. 
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m×p×ns` real array `v0`.   

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` can be used to specify the absolute tolerance in the gradient `∇J`, in infinity norm (default: `gtol = 1e-5`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
For stabilization purposes, the values `Jtol = 1.e-3`, `gtol = 1.e-2`, `maxit = 20` are used to favor faster convergence. 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic multipliers;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic multipliers;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 

A bound-constrained optimization can be performed using the keyword argument
`lub = (lb, ub)`, where `lb` is the lower bound and `ub` is the upper bound on the feedback gains. 
By default,  `lub = missing` in which case the bounds `lb = -Inf` and `ub = Inf` are assumed. 

_References_     

[1] A. Varga and S. Pieters. Gradient-based approach to solve optimal periodic output feedback problems. 
    Automatica, vol.34, pp. 477-481, 1998.

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
              Int. J. Control, vol, 67, pp, 69-87, 1997.

"""
function pdlqofc(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}; 
   S::Union{AbstractMatrix,PM3,Missing} = missing, sdeg = 1, G = I, lub = missing,
   vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
   optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), 
   maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false) where 
   {PM <: PeriodicMatrix, PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix}
   n = size(psys.A,2)
   mu = size(R,1)[1]; m = size(psys.B,2)[1]
   mu > m && throw(ArgumentError("R must have order at most $m"))
   nmax = maximum(n)  
   if isa(Q,PeriodicMatrix)
      issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic matrix"))
      lQ = length(Q.M); lA = length(psys.A.M)
      nQ = size(Q,1); 
      all([n[mod(i-1,lA)+1] == nQ[mod(i-1,lQ)+1] for i in 1:lcm(lQ,lA)]) || throw(ArgumentError("Q and A have incompatible dimensions"))
      Qa = convert(PeriodicArray,Q) 
   else
      issymmetric(Q) || throw(ArgumentError("Q must be a symmetric matrix"))
      nmax == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
      Qa = Q
   end
   if isa(R,PeriodicMatrix)
      issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
      all(mu .== size(R,2)) || throw(ArgumentError("R and B have incompatible dimensions"))
      Ra = convert(PeriodicArray,R) 
   else
      issymmetric(R) || throw(ArgumentError("R must be a symmetric matrix"))
      Ra = R
   end
   if ismissing(S)
      Sa = S
   else
      if isa(S,PeriodicMatrix)
         all(mu .== size(S.M,2)) || throw(ArgumentError("S and B have incompatible dimensions"))
         lS = length(S.M); lA = length(psys.A.M)
         mS = size(S,1); mA = size(psys.A,1)
         all([mA[mod(i-1,lA)+1] == mS[mod(i-1,lS)+1] for i in 1:lcm(lS,lA)]) || throw(ArgumentError("S and A have incompatible dimensions"))
         #size(S,1) == size(psys.A,1)  || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = convert(PeriodicArray,S) 
      else
         mu == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
         nmax == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = S
      end
   end

   _, info = pdlqofc(convert(PeriodicStateSpace{PeriodicArray},psys), Qa, Ra; S = Sa, vinit, sdeg, G, lub, optimizer, stabilizer, 
                     maxiter, vtol, Jtol, gtol, show_trace)
   return Kbuild(info.vopt,psys.D), info
end


function pdlqofc(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}; 
   S::Union{AbstractMatrix,PM3,Missing} = missing, sdeg = 1, G = I, lub = missing,
   vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
   optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), 
   maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false) where 
   {PM <: PeriodicArray, PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <:Union{PeriodicArray,PeriodicMatrix}}
   n = size(psys.A,2)
   p, m = size(psys); 
   mu = size(R,1); 
   mu > m && throw(ArgumentError("R must have order at most $m"))
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
   if !ismissing(S)
      mu == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
      n == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
   end

   sdeg <= 1 || throw(ArgumentError("desired stability degree must not exceed 1"))
   N = psys.A.dperiod*psys.A.nperiod

   ismissing(vinit) || (m,p,N) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($p,$m,$N)")) 
   ismissing(vinit) ? x = zeros(mu,p,N) : x = copy(vinit)
   A = psys.A; Bu = psys.B[:,1:mu]; C = psys.C; 
   F0 = PeriodicArray(x, A.period)
   stlim = 1-sqrt(eps())
   sdeg0 = maximum(abs.(pseig(A+Bu*F0*C)))
   if sdeg < 1 
      scal = 1/(sdeg^(1/N)); A = psys.A*scal; Bu = Bu*scal
      sd = maximum(abs.(pseig(A+B*F0*C)))
   else
      sd = sdeg0
   end 

   show_trace && println("initial stability degree = $sdeg0")

   T = eltype(A)
   Gt = similar(Array{T,3},n,n,N)
   Gt[:,:,N] = G == I ? Matrix{T}(I(n)) : G
   for i = 1:N-1
       Gt[:,:,i] = zeros(T,n,n)
   end
   GR = PeriodicArray(Gt, A.period)
   ismissing(lub) || (lb = fill(T(lub[1]),mu,p,N); ub = fill(T(lub[2]),mu,p,N))

   #nG =  (optimizer == NelderMead())
   nG = isa(stabilizer,NelderMead) && isa(optimizer,NelderMead)

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   WORK = (similar(Matrix{T},mu,n), similar(Matrix{T},mu,n), similar(Matrix{T},n,p))
   # WORK1 = (X, Y, At, xt, Q, pschur_ws)
   WORK1 = (Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(Gt) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   qr_ws = QRWs(zeros(8), zeros(4))
   WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
            Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
            qr_ws, QROrmWs(zeros(4), qr_ws.τ))

   Bu = psys.B[:,1:mu]
   if sd >= stlim
      nit = 10
      it = 1
      evs =  max(sd,1)
      while it <= nit && evs >= 1 
            #scal0 = max(evs*1.01,0.99)
            scal0 = evs*1.01
            scal = 1/(scal0^(1/N)); 
            As = scal*A; Bsu = scal*Bu; 
            par = (As, Bsu, C, R, Q, S, GR, WORK, WORK1, WORK2)
            if ismissing(lub)
               result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcfungrad!(F, G, x, par)), x, stabilizer,
                              Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-4, iterations = 20, show_trace=show_trace))
            else
               result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcfungrad!(F, G, x, par)), lb, ub, x, Fminbox(stabilizer),
                              Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-4, iterations = 20, show_trace=show_trace))
            end

            x = result.minimizer
            it = it+1
            KK = PeriodicArray(x, A.period)
            evs = maximum(abs.(pseig(psys.A+Bu*KK*C)))
      end
      it <= nit || error("no stabilizing initial feedback gain could be determined: Aborting")
   end
   if isnothing(optimizer)
      xopt = x
      sd = evs
      sd = sdeg < 0 ? maximum(abs.(pseig(psys.A+Bu*KK*C))) : evs
      par = (A, Bu, C, R, Q, S, GR, WORK, WORK1, WORK2)

      #fopt = nothing
      fopt = pdlqofcfungrad!(true, nothing, x, par)
      result = nothing
      Fopt = Kbuild(xopt,psys.D)
   else
      maxit = maxiter 
      par = (A, Bu, C, R, Q, S, GR, WORK, WORK1, WORK2)
      if ismissing(lub)
         result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcfungrad!(F, G, x, par)), x, optimizer,
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
      else
         result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcfungrad!(F, G, x, par)), lb, ub, x, Fminbox(optimizer),
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
      end
      xopt = result.minimizer
      KK = PeriodicArray(xopt, A.period)
      sd = maximum(abs.(pseig(psys.A+Bu*KK*C)))
      fopt = minimum(result)
      Fopt = Kbuild(xopt,psys.D)
   end
   show_trace && println("final stability degree = $sd")
   
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = sd, optres = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"

   return Fopt, info
end

function pdlqofcfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for pdlqofc
   (A, B, C, R, Q, S, GR, WORK, WORK1, WORK2) = par
   return fungrad!(Fun, Grad, x, A, B, C, R, Q, S, GR, WORK, WORK1, WORK2)
end
function fungrad!(Fun, Grad, x, A::PM, B::PM, C::PM, R, Q, S, GR::PM, WORK, WORK1, WORK2) where {PM <: PeriodicArray}
   F = PeriodicArray(x, A.period)
   FC = F*C
   RFC = R*FC
   Ar = A+B*FC
   if ismissing(S)
      #QR = Q + FC'*RFC
      QR = pmmultraddsym(Q,FC,RFC)
      ST = S
   else
      YY = S*FC; pmsymadd!(YY); 
      #QR = Q + FC'*RFC + YY
      QR = pmmultraddsym(Q+YY,FC,RFC)
      ST = isa(S,AbstractArray) ? PeriodicArray(Matrix(S'),A.period) : S' 
   end

   (X, Y, At, xt, QW, pschur_ws) = WORK1
   if isnothing(Grad)
      try
         copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
         pslyapd!(X, At, QR.M, xt, QW; adj = true, stability_check = true)
         return tr(X[:,:,1]*GR.M[:,:,end])
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   try
      copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
      pslyapd2!(Y, X, At, GR.M, QR.M, xt, QW, WORK2, pschur_ws; stability_check = true) 
   catch te
      isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
      return 1.e20
   end
   if !isnothing(Grad)
      #   Grad = 2*(B'*pmshift(X)*Ar + RFC + ST)*Y*C'
      gradeval!(Grad,Ar,B,C,RFC,X,Y,ST,WORK)
   end

   isnothing(Fun) ? (return nothing) : (return tr(X[:,:,1]*GR.M[:,:,end]))
end
function gradeval!(Grad::AbstractArray{T,3},A::PeriodicArray,B::PeriodicArray,C::PeriodicArray, RFC::PeriodicArray, X::AbstractArray{T,3}, Y::AbstractArray{T,3}, ST::Union{PeriodicArray,Missing}, WORK) where {T}
   m, p, N = size(Grad)
   n = size(A,2)
   pa = length(A) 
   pb = length(B)
   pc = length(C)
   prfc = length(RFC)  
   px = size(X,3)
   py = size(Y,3)
   ismissing(ST) || (pst = length(ST))
   (temp1,temp2,temp3) = WORK
   
   for i = 1:N
       ia = mod(i-1,pa)+1
       ib = mod(i-1,pb)+1
       ic = mod(i-1,pc)+1
       ix1 = mod(i,px)+1
       irfc = mod(i-1,prfc)+1
       iy = mod(i-1,py)+1

       # temp1 = B'*pmshift(X) (mxn)
       # temp2 = (B'*pmshift(X)*A + RFC + ST) (mxn)
       copyto!(temp2,view(RFC.M,:,:,irfc))  
       mul!(temp1,view(B.M,:,:,ib)',view(X,:,:,ix1))
       mul!(temp2,temp1,view(A.M,:,:,ia),1,1)
       ismissing(ST) || (is = mod(i-1,pst)+1; for ii = 1:m; for jj = 1:n; temp2[ii,jj] += ST.M[ii,jj,is]; end; end)

       # Grad = 2(B'*pmshift(X)*Ar + RFC + ST)*Y*C'
       mul!(temp3,view(Y,:,:,iy),view(C.M,:,:,ic)') # temp3 = Y*C' (n*p)
       mul!(view(Grad,:,:,i),temp2,temp3,2,0)
   end
end
function Kbuild(x::Array{T,3}, D::PM) where {T, PM <: PeriodicMatrix}
   K = PeriodicMatrix([x[:,:,i] for i in 1:size(x,3)], D.period) 
   return iszero(D) ? K : inv(I+K*D)*K
end
function Kbuild(x::Array{T,3}, D::PM) where {T, PM <: PeriodicArray}
   K = PeriodicArray(x, D.period) 
   return iszero(D) ? K : inv(I+K*D)*K
end
"""
    pdlqofc_sw(psys, Q, R, ns; S, vinit, kwargs...) -> (Fopt, info)

Compute for the discrete-time periodic state-space system `psys = (A(t),B(t),C(t),D(t))` of the form
   
   x(t+1) = A(t)x(t) + B(t)u(t) + Bw(t)w(t) 
     y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) ,

the optimal switching periodic feedback gain `Fopt(t)` in the output feedback control law  

    u(t) = Fopt(t)*y(t), 

which minimizes the expectation of the quadratic index 

     J = E{ Sum [x(t)'*Q(t)*x(t) + 2*x(t)'*S(t)*u(t) + u(t)'*R(t)*u(t)] },

where `Q(t)`, `R(t)` and `S(t)` are periodic weighting matrices. 
For a system of order `n` with `m` control inputs in `u(t)` and `p` measurable outputs in `y(t)`, 
`Q(t)` and `R(t)` are `n×n` and `m×m` symmetric periodic matrices, respectively, and 
`S(t)` is an `n×m` periodic matrix, which can be specified via the keyword argument `S`. 
By default `S = missing`, in which case, `S(t) = 0` is assumed.   
The periodic matrices `Q(t)`,`R(t)` and `S(t)` have the same type as the matrices of the state-space system,
i.e., either of type `PeriodicMatrix` or `PeriodicArray`.  
The dimension `m` of `u(t)` is deduced from the dimension of `R(t)`. 
`Q`, `R` and `S` can be alternatively provided as constant real matrices. 

The switching times for the resulting switching periodic gain `Fopt(t)` are specified by the 
`N`-dimensional integer vector `ns`. 
By default, `ns = 1:N`, where `N` is the maximal number of samples (i.e., `N = psys.period/psys.Ts`). 

The resulting `m×p` periodic output feedback gain `Fopt(t)` has the type `SwitchingPeriodicArray`
and is computed as `Fopt(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` 
defined as 

     F(t) = F_i for t ∈ [ns[i]Δ,ns[i+1]Δ) and i ∈ {1, ..., N-1}, or
     F(t) = F_N for t ∈ [ns[N]Δ,T),
           
where `T` is the system period (i.e., `T = psys.period`), `Δ` is the system sampling time (i.e., `Δ = psys.Ts`) 
and `F_i` is the `i`-th gain. 

If an initial value for the optimization variable `v` is provided using the keyword argument `vinit = gains`, then `gains`
must be an `m×p×N` array, where `m` and `p` are the numbers of system control inputs and outputs.
By default, `vinit = missing`, in which case a zero matrix `gains = 0` is assumed.  
   
The rest of keyword arguments contained in `kwargs` are the same as those used for the function [`pdlqofc`](@ref). 
See its documentation for the description of all keyword arguments. 

The computation of gradient fully exploits the switching structure of the feedback gain, using formulas which generalize
the case of constant feedback considered in [1].     

_References_     

[1] A. Varga and S. Pieters. Gradient-based approach to solve optimal periodic output feedback problems. 
    Automatica, vol.34, pp. 477-481, 1998.
"""
function pdlqofc_sw(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, ns = missing; 
   S::Union{AbstractMatrix,PM3,Missing} = missing, sdeg = 1, G = I, lub = missing, 
   vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
   optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), 
   maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false) where 
   {PM <: PeriodicArray, PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <:PeriodicArray}
   n = size(psys.A,2)
   p, m = size(psys); 
   mu = size(R,1); 
   mu > m && throw(ArgumentError("R must have order at most $m"))
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
   if !ismissing(S)
      m == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
      n == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
   end

   sdeg <= 1 || throw(ArgumentError("desired stability degree must not exceed 1"))
   N = psys.A.dperiod*psys.A.nperiod

   ismissing(ns) && (ns = collect(1:N))

   ps = length(ns)
   ns[1] > 0 || error("ns must have only strictly increasing positive values")
   for i in 1:ps-1
       ns[i+1] > ns[i] || error("ns must have only strictly increasing positive values")
   end

   if ismissing(vinit) 
      x = zeros(mu,p,ps)
   else
      (mu,p,ps) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($mu,$p,$ps)")) 
      x = copy(vinit)
   end

   A = psys.A; Bu = psys.B[:,1:mu]; C = psys.C; 
   #Ksw = SwitchingPeriodicArray(x, ns, A.period)
   F0 = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, A.period))
   stlim = 1-sqrt(eps()); 
   sdeg0 = maximum(abs.(pseig(A+Bu*F0*C)))
   if sdeg < 1 
      scal = 1/(sdeg^(1/N)); A = A*scal; Bu = Bu*scal
      sd = maximum(abs.(pseig(A+B*F0*C)))
   else
      sd = sdeg0
   end 

   T = eltype(A)
   Gt = similar(Array{T,3},n,n,N)
   Gt[:,:,N] = G == I ? Matrix{T}(I(n)) : G
   for i = 1:N-1
       Gt[:,:,i] = zeros(T,n,n)
   end
   GR = PeriodicArray(Gt, A.period)
   ismissing(lub) || (lb = fill(T(lub[1]),mu,p,N); ub = fill(T(lub[2]),mu,p,N))

   nG = isa(stabilizer,NelderMead) && isa(optimizer,NelderMead)

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   WORK = (similar(Matrix{T},mu,n), similar(Matrix{T},mu,n), similar(Matrix{T},n,p))
   # WORK1 = (WSGrad, X, Y, At, xt, Q, pschur_ws)
   WORK1 = (Array{T,3}(undef, mu, p, N), Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(Gt) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   qr_ws = QRWs(zeros(8), zeros(4))
   WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
            Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
            qr_ws, QROrmWs(zeros(4), qr_ws.τ))

   #evs = sd
   Bu = psys.B[:,1:mu]
   if sd >= stlim
      nit = 10
      it = 1
      evs = max(sd,1)
      while it <= nit && evs >= 1 
            scal0 = evs*1.01
            scal = 1/(scal0^(1/N)); 
            As = scal*A; Bsu = scal*Bu; 
            par = (As, Bsu, C, R, Q, S, ns, GR, WORK, WORK1, WORK2)
            if ismissing(lub)
               result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcswfungrad!(F, G, x, par)), x, stabilizer,
                              Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-4, iterations = 20, show_trace=show_trace))
            else
               result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcswfungrad!(F, G, x, par)), lb, ub, x, Fminbox(stabilizer),
                              Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-4, iterations = 20, show_trace=show_trace))
            end

            x = result.minimizer
            it = it+1
            KK = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, A.period))
            evs = maximum(abs.(pseig(psys.A+Bu*KK*C)))
      end
      it <= nit || error("no stabilizing initial feedback gain could be determined: Aborting")
      #show_trace && println("initial stability degree = $sdeg0")
   end
   if isnothing(optimizer)
      xopt = x
      sd = evs
      sd = sdeg < 0 ? maximum(abs.(pseig(psys.A+Bu*KK*C))) : evs
      par = (A, Bu, C, R, Q, S, ns, GR, WORK, WORK1, WORK2)

      #fopt = nothing
      fopt = pdlqofcswfungrad!(true, nothing, x, par)
      result = nothing
      Fopt = Kbuild_sw(x,psys.D,ns)
   else
      maxit = maxiter 
      par = (A, Bu, C, R, Q, S, ns, GR, WORK, WORK1, WORK2)
      if ismissing(lub)
         result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcswfungrad!(F, G, x, par)), x, optimizer,
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
      else
         result = optimize(Optim.only_fg!((F,G,x) -> pdlqofcswfungrad!(F, G, x, par)), lb, ub, x, Fminbox(optimizer),
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
      end
      xopt = result.minimizer
      #KK = PeriodicArray(xopt, A.period, ns)
      KK = convert(PeriodicArray,SwitchingPeriodicArray(xopt, ns, A.period))
      #KK = SwitchingPeriodicArray(x, ns, A.period)
      sd = maximum(abs.(pseig(psys.A+Bu*KK*C)))
      fopt = minimum(result)
      Fopt = Kbuild_sw(xopt,psys.D,ns)
   end
   show_trace && println("final stability degree = $sd")
   
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = sd, result = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"

   return Fopt, info
end
function pdlqofcswfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for pdlqofc
   (A, B, C, R, Q, S, ns, GR, WORK, WORK1, WORK2) = par
   return fungradsw!(Fun, Grad, x, A, B, C, R, Q, S, ns, GR, WORK, WORK1, WORK2)
end
function fungradsw!(Fun, Grad, x, A::PM, B::PM, C::PM, R, Q, S, ns, GR::PM, WORK, WORK1, WORK2) where {PM <: PeriodicArray}
   F = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, A.period))
   FC = F*C
   RFC = R*FC
   Ar = A+B*FC
   if ismissing(S)
      #QR = Q + FC'*RFC
      QR = pmmultraddsym(Q,FC,RFC)
      ST = S
   else
      YY = S*FC; pmsymadd!(YY); 
      #QR = Q + FC'*RFC + YY
      QR = pmmultraddsym(Q+YY,FC,RFC)
      ST = isa(S,AbstractArray) ? PeriodicArray(Matrix(S'),A.period) : S' 
   end

   (WSGrad, X, Y, At, xt, QW, pschur_ws) = WORK1
   if isnothing(Grad)
      try
         copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
         pslyapd!(X, At, QR.M, xt, QW; adj = true, stability_check = true)
         return tr(X[:,:,1]*GR.M[:,:,end])
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   try
      copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
      pslyapd2!(Y, X, At, GR.M, QR.M, xt, QW, WORK2, pschur_ws; stability_check = true) 
   catch te
      isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
      return 1.e20
   end
   if !isnothing(Grad)
      #   Grad = 2*(B'*pmshift(X)*Ar + RFC + ST)*Y*C'
      gradeval!(WSGrad,Ar,B,C,RFC,X,Y,ST,WORK)
      k = 0
      for i = 1:length(ns)
          copyto!(view(Grad,:,:,i), view(WSGrad,:,:,ns[i]))
          for j = 1:ns[i]-k-1
              Grad[:,:,i] += WSGrad[:,:,ns[i]-j]
          end
          k = ns[i]
      end
   end

   isnothing(Fun) ? (return nothing) : (return tr(X[:,:,1]*GR.M[:,:,end]))
end

function Kbuild_sw(x::Array{T,3}, D::PM, ns::Vector{Int}) where {T, PM <: PeriodicArray}
   K = SwitchingPeriodicArray(x, ns, D.period) 
   return iszero(D) ? K : inv(I+K*convert(SwitchingPeriodicArray,D))*K
end
function pdlqofc_sw(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, ns = missing;
                S::Union{AbstractMatrix,PM3,Missing} = missing, sdeg = 1, G = I, lub = missing,
                vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
                optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), 
                maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false) where 
                {PM <: PeriodicMatrix, PM1 <: PeriodicMatrix, PM2 <: PeriodicMatrix, PM3 <: PeriodicMatrix}
   n = size(psys.A,2)
   m = size(psys.B,2)[1]
   nmax = maximum(n)  
   if isa(Q,PeriodicMatrix)
      issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic matrix"))
      lQ = length(Q.M); lA = length(psys.A.M)
      nQ = size(Q,1); 
      all([n[mod(i-1,lA)+1] == nQ[mod(i-1,lQ)+1] for i in 1:lcm(lQ,lA)]) || throw(ArgumentError("Q and A have incompatible dimensions"))
      Qa = convert(PeriodicArray,Q) 
   else
      issymmetric(Q) || throw(ArgumentError("Q must be a symmetric matrix"))
      nmax == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
      Qa = Q
   end
   if isa(R,PeriodicMatrix)
      issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
      all(m .== size(R,2)) || throw(ArgumentError("R and B have incompatible dimensions"))
      Ra = convert(PeriodicArray,R) 
   else
      issymmetric(R) || throw(ArgumentError("R must be a symmetric matrix"))
      m == size(R,2) || throw(ArgumentError("Q and B have incompatible dimensions"))
      Ra = R
   end
   if ismissing(S)
      Sa = S
   else
      if isa(S,PeriodicMatrix)
         all(m .== size(S.M,2)) || throw(ArgumentError("S and B have incompatible dimensions"))
         lS = length(S.M); lA = length(psys.A.M)
         mS = size(S,1); mA = size(psys.A,1)
         all([mA[mod(i-1,lA)+1] == mS[mod(i-1,lS)+1] for i in 1:lcm(lS,lA)]) || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = convert(PeriodicArray,S) 
      else
         m == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
         nmax == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = S
      end
   end
   N = psys.A.dperiod*psys.A.nperiod
   ismissing(ns) && (ns = collect(1:N))

   _, info = pdlqofc_sw(convert(PeriodicStateSpace{PeriodicArray},psys), Qa, Ra, ns; S = Sa, sdeg, G, lub, optimizer, stabilizer, 
                        maxiter, vtol, Jtol, gtol, show_trace)
   return Kbuild_sw(info.vopt,psys.D,ns), info
end
function Kbuild_sw(x::Array{T,3}, D::PM, ns::Vector{Int}) where {T, PM <: PeriodicMatrix}
   K = SwitchingPeriodicMatrix([x[:,:,i] for i in 1:length(ns)], ns, D.period) 
   return iszero(D) ? K : inv(I+K*convert(SwitchingPeriodicMatrix,D))*K
end
"""
    pclqofc_sw(psys, Q, R, ts = missing; K = 1, sdeg = 0, G = I, vinit, optimizer, stabilizer,
               maxiter, vtol, Jtol, gtol, show_trace, solver, reltol, abstol, 
               N = 128, quad = false ) -> (Fopt,info)

Compute the optimal periodic stabilizing gain matrix `Fopt(t)`, such that for a continuous-time periodic state-space model 
`psys` of the form
   
      .
      x(t) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)  
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) ,
 
the output feedback control law

    u(t) = Fopt(t)*y(t), 
    
minimizes the expectation of the quadratic index

             ∞
     J = E{ Int [x(t)'*Q(t)*x(t) + u(t)'*R(t)*u(t)]dt },
            t=0

where `Q(t)` and `R(t)` are periodic weighting matrices. 
The matrices of the system `psys` are of type `PeriodicFunctionMatrix`. 
For a system of order `n` with `m` control inputs in `u(t)` and `p` measurable outputs in `y(t)`, 
`Q(t)` and `R(t)` are `n×n` and `m×m` symmetric periodic matrices of type `PeriodicFunctionMatrix`, respectively.                
The dimension `m` of `u(t)` is deduced from the dimension of `R(t)`. 
`Q` and `R` can be alternatively provided as constant real matrices. 

The resulting `m×p` periodic output feedback gain `Fopt(t)` is of type `PeriodicSwitchingMatrix`, 
with the switching times defined by the vector `ts`. 
The `ns` switching times contained in the vector `ts` must satisfy `0 = ts[1] < ts[2] < ... < ts[ns] < T`,
where `T` is the system period. 
If `ts = missing`, then `ts = [0]` is used by default (i.e., constant output feedback).

The output feedback gain `Fopt(t)` is computed as `Fopt(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` 
defined as 

     F(t) = F_i  for t ∈ [ts[i],ts[i+1]) and i ∈ {1, ..., ns-1} or 
     F(t) = F_ns for t ∈ [ts[ns],T)
           
where `F_i` is the `i`-th gain. 

The covariance matrix of the initial state `x(0)` can be specified via the keyword argument `G` (default: `G = I`)
and a desired stability degree of the closed-loop characteristic exponents can be specified using
the keyword argument `sdeg` (default: `sdeg = 0`). 

For the determination of the optimal feedback gains `F_i` for `i = 1, ...., ns` an optimization-based approach is employed using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-based limited-memory quasi-Newton method (also known as `L-BFGS`) for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`, where 
an initial step length for the line search algorithm is chosen using the keyword argument `alphaguess` 
(see the [`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl) package for alternative options). 
The employed default line search algorithm is `HagerZhang()` and an alternative method can be specified using the keyword argument `linesearch` 
(e.g., `linesearch = LineSearches.MoreThuente()`).  
Alternative gradient-based methods can be also selected, such as, for example,  the quasi-Newton method `BFGS` with `optimizer = Optim.BFGS()`, or 
for small size optimization problems, the Nelder-Mead gradient-free method with `optimizer = Optim.NelderMead()`. 
For the computation of the function `J` and its gradient  `∇J`, the formulas developed in [1] for stable systems are used. Each evaluation involves the solution of 
of a pair of periodic Lyapunov differential equations using single or multiple shooting methods proposed in [2].  
If the original system `psys` is unstable, the computation of a stabilizing feedback is performed using the same optimization techniques applied iteratively to systems 
with modified the state matrices of the form  `A(t)-αI`, where `α ≥ 0` is chosen such that `A(t)-αI` is stable, and the values of `α` are successively decreased until the stabilization is achieved.
The optimization method for stabilization can be independently selected using the keyword argument `stabilizer`, with the default setting  
`stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`. If only stabilization is desired, then use  `optimizer = nothing`. 

An internal optimization variable `v` is used, formed as an `m×p×ns` array with `v[:,:,i] := F_i`, for `i = 1, ..., ns`. 
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m×p×ns` array `v0`.   

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` can be used to specify the absolute tolerance in the gradient  `∇J`, in infinity norm (default: `gtol = 1e-5`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
For stabilization purposes,  the values `Jtol = 1.e-3`, `gtol = 1.e-2`, `maxit = 20` are used to favor faster convergence. 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 

Several keyword arguments can be used to control the integration of the involved ODE's for the solution of 
periodic differential Lyapunov equations for function and gradient evaluations. 

If `K = 1` (default), the single shooting method is employed to compute periodic generators [1]. 
If `K > 1`, the multiple-shooting method of [2] is employed, first, to convert the continuous-time periodic Lyapunov differential equations into discrete-time periodic Lyapunov equations satisfied by 
the generator solution in `K` grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [3]. If quad = true, a quadrature-based evaluation of gradients is used, as proposed in [1], in conjunction with
interpolation techniques. The number of sample values to be used for interpolation can be specified with the keyword parameter `N` (deafult: `N = 128`). 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt` (default: `dt = 0`, only used if `solver = "symplectic"`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable. 

[1] L. Vigano, M. Bergamasco, M. Lovera, and A. Varga. Optimal periodic output feedback control: a continuous-time approach and a case study.
    Int. J. Control, Vol. 83, pp. 897–914, 2010.  

[2] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[3] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
function pclqofc_sw(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
   ts = missing; K::Int = 1, sdeg::Real = 0, G = I, lub = missing, vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
   optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)),
   maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false,
   solver = "auto", reltol = 1.e-5, abstol = 1.e-7, N = 128, intpolmeth = "cubic", quad = false)  where 
   {PM <: PeriodicFunctionMatrix, PM1 <: PeriodicFunctionMatrix, PM2 <: PeriodicFunctionMatrix}
   period = psys.period
   #optimizer = Optim.NelderMead()
   n = size(psys.A,2)
   mu = size(R,1); m = size(psys.B,2)
   mu > m && throw(ArgumentError("R must have order at most $m"))
   p = size(psys.C,1)
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))

   K > 0 || throw(ArgumentError("number of discretization points must be positive"))
   N >= 32 || throw(ArgumentError("number of interpolation points must be at least 32, got $N"))
  
   sdeg <= 0 || throw(ArgumentError("desired stability degree must not exceed 0"))
   Δ = period/K
   δ = period/N

   #ismissing(ts) && (ts = collect(0:K-1)*Δ)
   ismissing(ts) && (ts = [0.])

   ns = length(ts)
   ts[1] == 0 || error("ts must have the first value equal to 0")
   for i in 1:ns
       Δti = i < ns ? ts[i+1] - ts[i] : period - ts[i]
       Δti > 0 || error("ts must have only strictly increasing positive values less than $period")
       check_commensurate_values(Δti,Δ) || 
           error("incommensurate switching times with the number of discretization points K = $K")
       #(quad && rationalize(Δti/δ).den !== 1) && 
       (quad && !check_commensurate_values(Δti,δ)) && 
           error("incommensurate switching times with the number of interpolation points N = $N")
   end

   if ismissing(vinit) 
      x = zeros(mu,p,ns)
   else
      (mu,p,ns) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($mu,$p,$ns)")) 
      x = copy(vinit)
   end

   A = psys.A; Bu = psys.B[:,1:mu]; C = psys.C; 
   T = eltype(A)
   X0 = G == I ? Matrix{T}(I(n)) : G
   #ismissing(lub) || (lb = fill(T(lub[1]),m*p*ps); ub = fill(T(lub[2]),m*p*ps)
   
   #Ksw = SwitchingPeriodicArray(x, ns, A.period)
   KK = convert(PeriodicFunctionMatrix,PeriodicSwitchingMatrix(x, ts, period))
   stlim = -sqrt(eps()); 
   sdeg0 = maximum(real(psceig(A+Bu*KK*C,100)))
   sdeg < 0 ? (A = A - sdeg*I; sd = maximum(real(psceig(A+Bu*KK*C,100)))) : sd = sdeg0

   show_trace && println("initial stability degree = $sdeg0")

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   #WORK = (similar(Matrix{T},m,n), similar(Matrix{T},m,n), similar(Matrix{T},n,p))
   # WORK1 = (WSGrad, X, Y, At, xt, Q, pschur_ws)
   #WORK1 = (Array{T,3}(undef, m, p, N), Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(n, N) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   #qr_ws = QRWs(zeros(8), zeros(4))
   # WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
   #          Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
   #          qr_ws, QROrmWs(zeros(4), qr_ws.τ))
   
   evs = sd
   if sd >= stlim
      shift = min(-sd*1.01,-0.001)
      options = (solver = solver, reltol = reltol, abstol = abstol, N = 128, intpolmeth = intpolmeth, quad = true)
      nit = 10
      it = 1
      evs = sd
      while it <= nit && evs >= 0 
            shift = min(-evs*1.01,-0.001)
            At = A+shift*I
            par = (K, At, Bu, C, R, Q, ts, X0, options)
            result = optimize(Optim.only_fg!((F,G,x) -> pclqofcswfungrad!(F, G, x, par)), x, stabilizer,
                Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-4, iterations = 20, show_trace=show_trace))

            x = result.minimizer
            it = it+1
            KK = PeriodicSwitchingMatrix(x, ts, period)
            evs = maximum(real.(psceig(A+Bu*convert(PeriodicFunctionMatrix,KK)*C,100)))
      end
      it <= nit || error("no stabilizing initial feedback gain could be determined: Aborting")
      #show_trace && println("initial stability degree = $sdeg0")
   end
   if isnothing(optimizer)
      xopt = x
      sd = evs
      sd = sdeg < 0 ? maximum(real(psceig(psys.A+Bu*convert(PeriodicFunctionMatrix,KK)*C,100))) : evs
      options = (solver = solver, reltol = reltol, abstol = abstol, N = N, intpolmeth = intpolmeth, quad = quad)
      par = (K, A, Bu, C, R, Q, ts, X0, options)
      fopt = pclqofcswfungrad!(true, nothing, x, par)
      result = nothing
      Fopt = Kbuild_sw(xopt,psys.D,ts)
   else
      maxit = maxiter 

      options = (solver = solver, reltol = reltol, abstol = abstol, N = N, intpolmeth = intpolmeth, quad = quad)
      par = (K, A, Bu, C, R, Q, ts, X0, options)
      result = optimize(Optim.only_fg!((F,G,x) -> pclqofcswfungrad!(F, G, x, par)), x, optimizer,
         Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   
      xopt = result.minimizer
      KK = PeriodicSwitchingMatrix(xopt, ts, period)
      sd = maximum(real.(psceig(A+Bu*convert(PeriodicFunctionMatrix,KK)*C,100)))
      fopt = minimum(result)
      Fopt = Kbuild_sw(xopt,psys.D,ts)
   end
   show_trace && println("final stability degree = $sd")
         
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = sd, optres = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"

   return Fopt, info
end
function check_commensurate_values(t1, t2)
      t = rationalize(t1/t2)
      t.den == 1 && (return true)
      tr = round(Int,t1/t2)
      tr == 0 && (return false)
      return tr ≈ t.num/t.den
end 
function pclqofcswfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for pdlqofc
   (K, A, B, C, R, Q, ts, X0, options) = par
   return fungradsw!(Fun, Grad, K, x, A, B, C, R, Q, ts, X0, options)
end

function fungradsw!(Fun, Grad, K, x, A::PM, B::PM, C::PM, R, Q, ts, X0, options) where {PM <: PeriodicFunctionMatrix}
   period = A.period
   F = convert(PeriodicFunctionMatrix,PeriodicSwitchingMatrix(x, ts, period))
   FC = F*C
   Ar = A+B*FC
   RFC = R*FC
   QR = pmmultraddsym(Q,FC,RFC)  
   quad = options.quad

   #(WSGrad, X, Y, At, xt, QW, pschur_ws) = WORK1
   #K = 100
   if isnothing(Grad)
      try
         P = pgclyap(Ar, QR, K; adj = true, solver = options.solver, reltol = options.reltol, abstol = options.abstol, stability_check = true)
         return tr(P(0)*X0)
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   Z, Y = try
      pgclyap2(Ar, X0, QR, K; solver = options.solver, reltol = options.reltol, abstol = options.abstol, stability_check = true)
   catch te
      isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
      return 1.e20
   end
   if !isnothing(Grad)
      #   Grad = 2*(B'*pmshift(X)*Ar + RFC + ST)*Y*C'
      if K >= 32
         P = convert(PeriodicFunctionMatrix, Y, method = options.intpolmeth)
         #quad && (X = convert(PeriodicFunctionMatrix, Z, method = options.intpolmeth))
         if quad 
            #X = convert(PeriodicFunctionMatrix, Z, method = options.intpolmeth)
            Xts = Z.values
            Xts = [Xts; [Xts[1]-X0]]
            X = PeriodicSystems.ts2fm(Xts, period; method = options.intpolmeth)
         end
      else
         N = max(options.N,32)
         Yt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y, Ar, QR; adj = true, solver = options.solver, reltol = options.reltol, abstol = options.abstol),A.period)
         P = convert(PeriodicFunctionMatrix,convert(PeriodicTimeSeriesMatrix,Yt;ns=N), method = options.intpolmeth)
         if quad   
            #Xt = PeriodicFunctionMatrix(t->PeriodicSystems.tvclyap_eval(t, Z, Ar, Z(0); adj = false, solver = options.solver, reltol = options.reltol, abstol = options.abstol),A.period)
            Xt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Z, Ar; solver = options.solver, reltol = options.reltol, abstol = options.abstol),A.period)
            Xts = Xt.f.((0:N-1)*period/N)
            Xts = [Xts; [Xts[1]-X0]]
            #X = convert(PeriodicFunctionMatrix,convert(PeriodicTimeSeriesMatrix,Xt;ns=N), method = options.intpolmeth)
            X = PeriodicSystems.ts2fm(Xts, period; method = options.intpolmeth)
         end
      end
      quad || (V = MatrixEquations.triu2vec(Z(0)))
      Ns = length(ts)
      for i = 1:Ns
          t0 = ts[i]; tf = i < Ns ? ts[i+1] : A.period   
          if quad
             Grad[:,:,i] .= QuadGK.quadgk(t-> 2*(B(t)'*P(t)+RFC(t))*X(t)*C(t)', t0, tf, rtol=1e-5)[1]
          else    
             Grad[:,:,i] = tvgrad!(V, Ar, B, C, RFC, P, tf, t0; solver = options.solver, reltol = options.reltol, abstol = options.abstol) 
          end
      end
   end

   isnothing(Fun) ? (return nothing) : (tr(P(0)*X0))
end
Kbuild_sw(x::AbstractArray{T,3}, psys::PeriodicStateSpace{PM}, ts) where {T <: Real, PM <: PeriodicFunctionMatrix} = Kbuild_sw(x,psys.D,ts)
function Kbuild_sw(x::AbstractArray{T,3}, D::PM, ts) where {T, PM <: PeriodicFunctionMatrix}
   K = PeriodicSwitchingMatrix(x, ts, D.period)
   mu = size(x,1)
   return iszero(D) ? K : inv(I+K*convert(PeriodicSwitchingMatrix,D[:,1:mu]))*K
end

function tvgrad!(V, A::PM, B::PM, C::PM, RFC::PM, P::PFM, tf, t0; solver = options.solver, reltol = options.reltol, abstol = options.atol) where 
   {PM <: Union{PeriodicFunctionMatrix,HarmonicArray}, PFM <: PeriodicFunctionMatrix}
   """
   tvgrad!(A, B, C, R, F, P, V, tf, to; solver, reltol, abstol) ->  G 

   Compute the gradient of the objective function for a linear-quadratic problem for a closed-loop system (A(t),B(t),C(t),0) with output feedback. 
   by integrating for tf > t0, jointly the differential matrix Lyapunov equation
      
              . 
              X(t) = A(t)*X(t)+X(t)*A'(t), X(t0) = V
      
   and
      
              .
              G(t) = 2*(B'(t)*P(t)+R(t)*F*C(t))*X(t)*C'(t),  G(t0) = 0;
      
      
   The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   Depending on the desired relative accuracy `reltol`, 
   lower order solvers are employed for `reltol >= 1.e-4`, 
   which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
   higher order solvers are employed able to cope with high accuracy demands. 

   The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

   `solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

   `solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

   `solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
   """    
    n = size(A,1)   
    n == size(A,2) || error("the periodic matrix A must be square")
    n == size(C,2) || error("the periodic matrix C must have same number of columns as A")
    n == size(B,1) || error("the periodic matrix B must have same number of rows as A")
    np = size(B,2)*size(C,1)
    nv = length(V)
    T = promote_type(typeof(t0), typeof(tf))
    # using OrdinaryDiffEq
    u0 = [zeros(T,np); V]
    tspan = (T(t0),T(tf))
    fgrad1!(du,u,par,t) = muladdcsymgr!(du, u, A(t), B(t), C(t), RFC(t), P(t)) 
    prob = ODEProblem(fgrad1!, u0, tspan)
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
    else 
       if reltol > 1.e-4  
          # low accuracy automatic selection
          sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
       else
          # high accuracy automatic selection
          sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
       end
    end
    copyto!(V,sol(tf)[np+1:np+nv])
    return sol(tf)[1:np] 
end
function muladdcsymgr!(y::AbstractVector, x::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, RFC::AbstractMatrix, P::AbstractMatrix)
   # [2*(B'*P+RFC)*X*C']
   # [triu(A*X + X*A')]
   n = size(A,1)
   lx = div(n*(n+1),2)
   k = length(x)-lx
   T1 = promote_type(eltype(A), eltype(x))
   # TO DO: eliminate building of X by using directly x
   #X = MatrixEquations.vec2triu(convert(AbstractVector{T1}, view(x,1:length(x)-1)), her=true)
   X = MatrixEquations.vec2triu(convert(AbstractVector{T1}, view(x,k+1:k+lx)), her=true)
   #copyto!(view(y,1:k), 2*(B'*P+R*F*C)*X*C') 
   copyto!(view(y,1:k), 2*(B'*P+RFC)*X*C') 
   copyto!(view(y,k+1:k+lx),MatrixEquations.triu2vec(A*X+X*A'))
   # @inbounds begin
   #    k =+ 1
   #    for j = 1:n
   #       for i = 1:j
   #           temp = zero(T1)
   #           for l = 1:n
   #               temp += A[i,l] * X[l,j] + X[i,l] * A[j,l]
   #           end
   #           y[k] = temp
   #           k += 1
   #       end
   #    end
   # end
   #@show norm(y[end-lx+1:end]-MatrixEquations.triu2vec(A*X+X*A'))
   return y
end
"""
    pclqofc_hr(psys, Q, R, nh = 0; K = 1, sdeg = 0, G = I, vinit, optimizer, stabilizer,
               maxiter, vtol, Jtol, gtol, show_trace, solver, reltol, abstol, 
               N = 128, quad = false ) -> (Fopt,info)

Compute the optimal periodic stabilizing gain matrix `Fopt(t)`, such that for a continuous-time periodic state-space model 
`psys` of the form
   
      .
      x(t) = A(t)x(t) + B(t)u(t) + Bw(t)w(t)  
      y(t) = C(t)x(t) + D(t)u(t) + Dw(t)w(t) ,
 
the output feedback control law

    u(t) = Fopt(t)*y(t), 
    
minimizes the expectation of the quadratic index

             ∞
     J = E{ Int [x(t)'*Q(t)*x(t) + u(t)'*R(t)*u(t)]dt },
            t=0

where `Q(t)` and `R(t)` are periodic weighting matrices. 
The matrices of the system `psys` are of type `HarmonicArray`. 
For a system of order `n` with `m` control inputs in `u(t)` and `p` measurable outputs in `y(t)`, 
`Q(t)` and `R(t)` are `n×n` and `m×m` symmetric periodic matrices of type `HarmonicArray`, respectively.                
The dimension `m` of `u(t)` is deduced from the dimension of `R(t)`. 
`Q` and `R` can be alternatively provided as constant real matrices. 

The resulting `m×p` periodic output feedback gain `Fopt(t)` is of type `HarmonicArray` and
is computed as `Fopt(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` in the harmonic representation form 

                  nh
     F(t) = F0 +  ∑ ( Fc_i*cos(i*t*2*π/T)+Fs_i*sin(i*2*π*t/T) ) ,
                 i=1 

where `T` is the system period, `F0` is the constant term, `Fc_i` is the `i`-th cosinus coefficient matrix and `Fs_i` is the `i`-th sinus coefficient matrix. 
By default, the number of harmonics is `nh = 0` (i.e., constant output feedback is used).

The covariance matrix of the initial state `x(0)` can be specified via the keyword argument `G` (default: `G = I`)
and a desired stability degree of the closed-loop characteristic exponents can be specified using
the keyword argument `sdeg` (default: `sdeg = 0`). 

For the determination of the optimal feedback gains `F0`, `Fc_i` and `Fs_i` for `i = 1, ...., nh` an optimization-based approach is employed using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-based limited-memory quasi-Newton method (also known as `L-BFGS`) for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`, where 
an initial step length for the line search algorithm is chosen using the keyword argument `alphaguess` 
(see the [`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl) package for alternative options). 
The employed default line search algorithm is `HagerZhang()` and an alternative method can be specified using the keyword argument `linesearch` 
(e.g., `linesearch = LineSearches.MoreThuente()`).  
Alternative gradient-based methods can be also selected, such as, for example,  the quasi-Newton method `BFGS` with `optimizer = Optim.BFGS()`, or 
for small size optimization problems, the Nelder-Mead gradient-free method with `optimizer = Optim.NelderMead()`. 
For the computation of the function `J` and its gradient  `∇J`, the formulas developed in [1] for stable systems are used. Each evaluation involves the solution of 
of a pair of periodic Lyapunov differential equations using single or multiple shooting methods proposed in [2].  
If the original system `psys` is unstable, the computation of a stabilizing feedback is performed using the same optimization techniques applied iteratively to systems 
with modified the state matrices of the form  `A(t)-αI`, where `α ≥ 0` is chosen such that `A(t)-αI` is stable, and the values of `α` are successively decreased until the stabilization is achieved.
The optimization method for stabilization can be independently selected using the keyword argument `stabilizer`, with the default setting  
`stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true))`. If only stabilization is desired, then use  `optimizer = nothing`. 

An internal optimization variable `v` is used, formed as an `m*p*(2*nh+1)` dimensional vector `v := [vec(F0); vec(Fc_1); vec(Fs_1), ... ; vec(Kc_nh); vec(Ks_nh)]'. 
By default, `v` is initialized as `v = 0` (i.e., a zero vector of appropriate dimension). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary vector `v0`.   

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` can be used to specify the absolute tolerance in the gradient  `∇J`, in infinity norm (default: `gtol = 1e-5`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
For stabilization purposes,  the values `Jtol = 1.e-3`, `gtol = 1.e-2`, `maxit = 20` are used to favor faster convergence. 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 

Several keyword arguments can be used to control the integration of the involved ODE's for the solution of 
periodic differential Lyapunov equations for function and gradient evaluations. 

If `K = 1` (default), the single shooting method is employed to compute periodic generators [1]. 
If `K > 1`, the multiple-shooting method of [2] is employed, first, to convert the continuous-time periodic Lyapunov differential equations into discrete-time periodic Lyapunov equations satisfied by 
the generator solution in `K` grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [3]. If quad = true, a quadrature-based evaluation of gradients is used, as proposed in [1], in conjunction with
interpolation techniques. The number of sample values to be used for interpolation can be specified with the keyword parameter `N` (deafult: `N = 128`). 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt` (default: `dt = 0`, only used if `solver = "symplectic"`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable. 

[1] L. Vigano, M. Bergamasco, M. Lovera, and A. Varga. Optimal periodic output feedback control: a continuous-time approach and a case study.
    Int. J. Control, Vol. 83, pp. 897–914, 2010.  

[2] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[3] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
"""
function pclqofc_hr(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
   nh::Int = 0; K::Int = 1, sdeg::Real = 0, G = I, vinit::Union{AbstractVector{<:Real},Missing} = missing, 
   optimizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)), stabilizer = LBFGS(;alphaguess = LineSearches.InitialStatic(;scaled=true)),
   maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false,
   solver = "auto", reltol = 1.e-5, abstol = 1.e-7, N = 128, intpolmeth = "cubic", quad = false)  where 
   {PM <: HarmonicArray, PM1 <: HarmonicArray, PM2 <: HarmonicArray}
   period = psys.period
   n = size(psys.A,2)
   mu = size(R,1); m = size(psys.B,2)
   mu > m && throw(ArgumentError("R must have order at most $m"))
   p = size(psys.C,1)
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
   nh >= 0 || throw(ArgumentError("number of harmonics must be non-negative"))
   K > 0 || throw(ArgumentError("number of discretization points must be positive"))
   
   A = psys.A; Bu = psys.B[:,1:mu]; 
   T = eltype(A)
 
   nh1 = nh+1
   pt = p*(2nh+1)
   
   if nh > 0
      H = zeros(Complex{T},pt,p,nh1)
      copyto!(view(H,1:p,:,1),I)
      k = p
      for i = 2:nh1
         copyto!(view(H,k+1:k+p,:,i),I)
         k += p
         copyto!(view(H,k+1:k+p,:,i),im*I)
         k += p
      end
      C = HarmonicArray(H,period)*psys.C
   else
      C = psys.C
   end



   sdeg <= 0 || throw(ArgumentError("desired stability degree must not exceed 0"))

   lx = mu*pt
   if ismissing(vinit) 
      x = zeros(lx)
   else
      length(vinit) == lx || throw(ArgumentError("length of vinit must be $lx")) 
      x = copy(vinit)
   end

   X0 = G == I ? Matrix{T}(I(n)) : G

   KK = HarmonicArray(reshape(x,mu,pt),period)
   stlim = -sqrt(eps()); 
   sdeg0 = maximum(real(psceig(A+Bu*KK*C,100)))
   sdeg < 0 ? (A = A - sdeg*I; sd = maximum(real(psceig(A+Bu*KK*C,100)))) : sd = sdeg0

   show_trace && println("initial stability degree = $sdeg0")

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   #WORK = (similar(Matrix{T},m,n), similar(Matrix{T},m,n), similar(Matrix{T},n,p))
   # WORK1 = (WSGrad, X, Y, At, xt, Q, pschur_ws)
   #WORK1 = (Array{T,3}(undef, m, p, N), Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(n, N) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   #qr_ws = QRWs(zeros(8), zeros(4))
   # WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
   #          Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
   #          qr_ws, QROrmWs(zeros(4), qr_ws.τ))
   
   if sd >= stlim
      shift = min(-sd*1.01,-0.001)
      options = (solver = solver, reltol = reltol, abstol = abstol, N = 128, intpolmeth = intpolmeth, quad = true)
      nit = 10
      it = 1
      evs = max(sd,0)
      while it <= nit && evs >= 0 
            shift = min(-evs*1.01,-0.001)
            At = A+shift*I
            par = (K, At, Bu, C, R, Q, X0, options)
            result = optimize(Optim.only_fg!((F,G,x) -> pclqofchrfungrad!(F, G, x, par)), x, stabilizer,
                Optim.Options(x_tol = vtol, f_tol = 1.e-3, g_tol = 1.e-2, iterations = 20, show_trace=show_trace))

            x = result.minimizer
            it = it+1
            KK = HarmonicArray(reshape(x,mu,pt),period)
            evs = maximum(real.(psceig(A+Bu*KK*C,100)))
      end
      it <= nit || error("no stabilizing initial feedback gain could be determined: Aborting")
      show_trace && println("initial stability degree = $sdeg0")
   end
   if isnothing(optimizer)
      xopt = x
      sd = evs
      sd = sdeg < 0 ? maximum(real(psceig(psys.A+Bu*KK*C,100))) : evs
      options = (solver = solver, reltol = reltol, abstol = abstol, N = N, intpolmeth = intpolmeth, quad = quad)
      par = (K, A, Bu, C, R, Q, X0, options)

      #fopt = nothing
      fopt = pclqofchrfungrad!(true, nothing, x, par)
      result = nothing
      Fopt = Fbuild_hr(xopt,psys.D,nh)
   else
      maxit = maxiter 

      options = (solver = solver, reltol = reltol, abstol = abstol, N = N, intpolmeth = intpolmeth, quad = quad)
      par = (K, A, Bu, C, R, Q, X0, options)
      result = optimize(Optim.only_fg!((F,G,x) -> pclqofchrfungrad!(F, G, x, par)), x, optimizer,
         Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   
      xopt = result.minimizer
      KK = HarmonicArray(reshape(xopt,mu,pt),period)
      sd = maximum(real.(psceig(psys.A+Bu*KK*C,100)))
      fopt = minimum(result)
      Fopt = Fbuild_hr(xopt,psys.D,nh)
   end
      
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = sd, optres = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"

   return Fopt, info
end
function pclqofchrfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for pdlqofc
   (K, A, B, C, R, Q, X0, options) = par
   return fungradhr!(Fun, Grad, K, x, A, B, C, R, Q, X0, options)
end

function fungradhr!(Fun, Grad, K, x, A::PM, B::PM, C::PM, R, Q, X0, options) where {PM <: HarmonicArray}
   n, m, pt = size(A,1), size(B,2), size(C,1)
   period = A.period
   F = HarmonicArray(reshape(x,m,pt),period)
   FC = F*C
   Ar = A+B*FC
   RFC = R*FC
   QR = pmmultraddsym(Q,FC,RFC)  
   quad = options.quad

   #(WSGrad, X, Y, At, xt, QW, pschur_ws) = WORK1
   if isnothing(Grad)
      try
         P = pgclyap(Ar, QR, K; adj = true, solver = options.solver, reltol = options.reltol, abstol = options.abstol, stability_check = true)
         return tr(P(0)*X0)
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   Z, Y = try
      pgclyap2(Ar, X0, QR, K; solver = options.solver, reltol = options.reltol, abstol = options.abstol, stability_check = true)
   catch te
      isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
      return 1.e20
   end
   if !isnothing(Grad)
      #   Grad = 2*(B'*pmshift(X)*Ar + RFC + ST)*Y*C'
      if K >= 32
         P = convert(PeriodicFunctionMatrix, Y, method = options.intpolmeth)
         #P = PeriodicMatrixEquations.pclyap_intpol(Ar, QR, Y; N = 1, adj = true, solver = options.solver, reltol = options.reltol, abstol = options.abstol)        
         if quad 
            Xts = Z.values
            Xts = [Xts; [Xts[1]-X0]]
            X = PeriodicSystems.ts2fm(Xts, period; method = options.intpolmeth)
         end
      else
         N = max(options.N,32)
         Yt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y, Ar, QR; adj = true, solver = options.solver, reltol = options.reltol, abstol = options.abstol),period)
         P = convert(PeriodicFunctionMatrix,convert(PeriodicTimeSeriesMatrix,Yt;ns=N), method = options.intpolmeth)
         if quad 
            #Xt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Z, Ar, Z(0); adj = false, solver = options.solver, reltol = options.reltol, abstol = options.abstol),A.period)
            Xt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Z, Ar; solver = options.solver, reltol = options.reltol, abstol = options.abstol),A.period)
            Xts = Xt.f.((0:N-1)*period/N)
            Xts = [Xts; [Xts[1]-X0]]
            X = PeriodicSystems.ts2fm(Xts, period; method = options.intpolmeth)
         end
      end
      quad || (V = MatrixEquations.triu2vec(Z(0)))
      t0 = 0; tf = A.period   
      if quad
         Grad[:] .= QuadGK.quadgk(t-> 2*(B(t)'*P(t)+RFC(t))*X(t)*C(t)', t0, tf)[1][:]
      else    
         Grad[:] .= tvgrad!(V, Ar, B, C, RFC, P, tf, t0; solver = options.solver, reltol = options.reltol, abstol = options.abstol)[:]
      end
   end

   isnothing(Fun) ? (return nothing) : (tr(P(0)*X0))
end
Fbuild_hr(x::AbstractVector{T}, psys::PeriodicStateSpace{PM}, nh::Int; PFM = false) where {T <: Real, PM <: HarmonicArray} = Fbuild_hr(x,psys.D,nh;PFM)
function Fbuild_hr(x::AbstractVector{T}, D::PM, nh::Int; PFM = false) where {T <: Real, PM <: HarmonicArray}
   period = D.period
   p = size(D,1)
   mu = div(length(x),p*(2nh+1))
   pm = p*mu
   khr = Array{Complex{T},3}(undef, mu, p, nh+1)
   copyto!(view(khr,1:mu,1:p,1:1),complex.(reshape(view(x,1:pm),mu,p)))
   ks = 0
   for i = 1:nh
      kc = ks+pm; ks = kc + pm
      copyto!(view(khr,1:mu,1:p,i+1), complex.(view(x,kc+1:ks),view(x,ks+1:ks+pm)))
   end
   K = HarmonicArray{:c,T}(khr, period, 1)

   if PFM
      K = convert(PeriodicFunctionMatrix, K)
      Du = convert(PeriodicFunctionMatrix, D[:,1:mu])
   else
      Du = D[:,1:mu]
   end

   if iszero(D)
      return K
   else
      G = I+K*D[:,1:mu]
      f = mu == 1 ? t->norm(G(t)) : t-> 1/cond(G(t))
      res = optimize(f,0,period,Optim.Brent(),rel_tol = eps())
      res.minimum > 1.e-8 || (@warn "possible unbounded feedback gain near t = $(res.minimizer)")
      return inv(G)*K
   end
end

"""
    pcpofstab_sw(psys, ts = missing; K = 100, vinit, optimizer, maxit, vtol, Jtol, gtol, show_trace) -> (Fstab,info)

For a continuous-time periodic system `psys = (A(t), B(t), C(t), D(t))` of period `T` determine a periodic output feedback gain matrix 
`Fstab(t)` of the same period and switching times `ts`,  
such that the characteristic exponents `Λ` of the closed-loop state-matrix `A(t)+B(t)*Fstab(t)*inv(I-D(t)*Fstab(t))*C(t)` are stable. 
The matrices of the system `psys` are of type `PeriodicFunctionMatrix`. 
The `ns` switching times contained in the vector `ts` must satisfy `0 = ts[1] < ts[2] < ... < ts[ns] < T`. 
If `ts = missing`, then `ts = [0]` is used by default (i.e., constant output feedback). 

The output feedback gain `Fstab(t)` is computed as `Fstab(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` 
defined as 

     F(t) = F_i  for t ∈ [ts[i],ts[i+1]) and i ∈ {1, ..., ns-1} or 
     F(t) = F_ns for t ∈ [ts[ns],T)
           
where `F_i` is the `i`-th gain.  
The resulting periodic matrix `Fstab(t)` is of type `PeriodicSwitchingMatrix`.
The corresponding closed-loop periodic system can be obtained using the function [`psfeedback`](@ref).

For the determination of the optimal feedback gains `F_i` for `i = 1, ...., ns` an optimization-based approach is employed using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-free _Nelder-Mead_ local search method for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = Optim.NelderMead()`.   
The alternative gradient-free _Simulated Annealing_ global search method can be selected with
`optimizer = Optim.SimulatedAnnealing()`. 

For a system with `m` inputs and `p` outputs, 
an internal optimization variable `v` is used, formed as an `m×p×ns` array with `v[:,:,i] := F_i`, for `i = 1, ..., ns`. 
The performance index to be minimized is `J := sdeg(v)`, 
where `sdeg(v)` is the stability degree defined as the largest real part of the characteristic exponents 
of `Af(t) := A(t)+B(t)*F(t)*C(t)`. The keyword argument `K` is the number of factors used to express the monodromy matrix of `Af(t)` (default: `K = 100`), 
when evaluating the characteristic exponents.   
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m×p×ns` array `v0`.  

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` is the method specific main convergence tolerance (default: `gtol = 1e-3`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
(see the documentation of the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package for additional information). 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 
"""
function pcpofstab_sw(psys::PeriodicStateSpace{PM}, ts::Union{AbstractVector{<:Real},Missing} = missing; K = 100, vinit::Union{AbstractArray{T,3},Missing} = missing, 
                      optimizer = NelderMead(), maxit = 1000, vtol = 0., Jtol = 0., gtol = 1e-3, show_trace = false) where {T <: Real, PM <: PeriodicFunctionMatrix}
   period = psys.period
   p, m = size(psys)
   Δ = period/K
   if ismissing(ts) 
      ts = [0.]; ns = 1
   else
      ns = length(ts)
      ts[1] == 0 || error("ts must have the first value equal to 0")
      for i in 1:ns
          Δti = i < ns ? ts[i+1] - ts[i] : period - ts[i]
          Δti > 0 || error("ts must have only strictly increasing positive values less than $period")
          if Δ < Δti
            check_commensurate_values(Δti,Δ) || 
              error("incommensurate switching times with the discretization points")
          else
             check_commensurate_values(Δ,Δti) || 
               error("incommensurate switching times with the system period")
          end
      end
   end

   if ismissing(vinit) 
      x = zeros(m,p,ns)
   else
      (m,p,ns) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($m,$p,$ns)")) 
      x = copy(vinit)
   end

   A = psys.A; B = psys.B; C = psys.C; 
   KK = convert(PeriodicFunctionMatrix,PeriodicSwitchingMatrix(x, ts, period))
   smarg = -sqrt(eps()); 
   sdeg0 = maximum(real(psceig(A+B*KK*C,K)))
   show_trace && println("initial stability degree = $sdeg0")

   result = optimize(x->pssdeg_sw(x,K,psys,ts), x, optimizer, 
       Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   xopt = result.minimizer
   fopt = minimum(result)
   fopt >= sdeg0  && (@warn "no improvement of stability degree achieved")
      
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = fopt, optres = result)
   Fopt = Kbuild_sw(xopt,psys.D,ts)
   fopt < smarg || @warn "no stabilization achieved: increase the number of time values"

   return Fopt, info

end
function pssdeg_sw(x::AbstractArray{T,3}, K, psys::PeriodicStateSpace{PM}, ts::AbstractVector{<:Real}) where  {T <: Real, PM <: PeriodicFunctionMatrix}
   KK = PeriodicSwitchingMatrix(x, ts, psys.period)
   return maximum(real(psceig(psys.A+psys.B*KK*psys.C,K)))
end
"""
    pcpofstab_hr(psys,  nh = 0; K = 100, vinit, optimizer = "local", maxiter, vtol, Jtol, gtol, show_trace) -> (Fstab,info)

For a continuoous-time periodic system `psys = (A(t), B(t), C(t), D(t))` of period `T` determine a periodic output feedback gain matrix 
`Fstab(t)` of the same period,  
such that the characteristic exponents `Λ` of the closed-loop state-matrix `A(t)+B(t)*Fstab(t)*inv(I-D(t)*Fstab(t))*C(t)` are stable. 
The matrices of the system `psys` are of type `HarmonicArray`. 

The resulting output feedback gain `Fstab(t)` is computed as `Fstab(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` in the harmonic representation form 

                  nh
     F(t) = F0 +  ∑ ( Fc_i*cos(i*t*2*π/T)+Fs_i*sin(i*2*π*t/T) ) ,
                 i=1 

where `F0` is the constant term, `Fc_i` is the `i`-th cosinus coefficient matrix and `Fs_i` is the `i`-th sinus coefficient matrix. 
By default, the number of harmonics is `nh = 0` (i.e., constant output feedback is used).
The resulting periodic matrix `Fstab(t)` is of type `HarmonicArray`.
The corresponding closed-loop periodic system can be obtained using the function [`psfeedback`](@ref).

For the determination of the optimal feedback gains `F0`, `Fc_i` and `Fs_i` for `i = 1, ...., nh` 
an optimization-based approach is employed using using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-free _Nelder-Mead_ local search method for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = Optim.NelderMead()`.   
The alternative gradient-free _Simulated Annealing_ global search method can be selected with 
`optimizer = Optim.SimulatedAnnealing()`. 

For a system with `m` inputs and `p` outputs, 
an internal optimization variable `v` is used, formed as an `m*p*(2*nh+1)` dimensional vector 
`v := [vec(F0); vec(Fc_1); vec(Fs_1), ... ; vec(Fc_nh); vec(Fs_nh)]`. 
The performance index to be minimized is `J := sdeg(v)`, 
where `sdeg(v)` is the stability degree defined as the largest real part of the characteristic exponents 
of `Af(t) := A(t)+B(t)*F(t)*C(t)`. The keyword argument `K` is the number of factors used to express the monodromy matrix of `Af(t)` (default: `K = 100`), 
when evaluating the characteristic exponents.   
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m*p*(2*nh+1)` array `v0`.  

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` is the method specific main convergence tolerance (default: `gtol = 1e-3`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
(see the documentation of the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package for additional information). 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 
"""
function pcpofstab_hr(psys::PeriodicStateSpace{PM}, nh::Int = 0; K = 100, vinit::Union{AbstractVector{<:Real},Missing} = missing, 
                      optimizer = NelderMead(), maxit = 1000, vtol = 0., Jtol = 0., gtol = 1e-3, show_trace = false) where {PM <: HarmonicArray}
   period = psys.period
   p, m = size(psys)

   A = psys.A; B = psys.B; C = psys.C; 
   T = eltype(A)
 
   nh1 = nh+1
   pt = p*(2nh+1)

   lx = m*pt
   if ismissing(vinit) 
      x = zeros(lx)
   else
      length(vinit) == lx || throw(ArgumentError("length of vinit must be $lx")) 
      x = copy(vinit)
   end


   smarg = -sqrt(eps()); 
   sdeg0 = pssdeg_hr(x,K,psys,nh)
   show_trace && println("initial stability degree = $sdeg0")

   result = optimize(x->pssdeg_hr(x,K,psys,nh), x, optimizer, 
       Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   xopt = result.minimizer
   fopt = minimum(result)
   fopt >= sdeg0  && (@warn "no improvement of stability degree achieved")
      
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = fopt, optres = result)
   Fopt = Fbuild_hr(xopt,psys.D,nh)
   fopt < smarg || @warn "no stabilization achieved: increase the number of time values"

   return Fopt, info

end
function pssdeg_hr(x::AbstractVector{T}, K::Int, psys::PeriodicStateSpace{PM}, nh::Int) where {T <: Real, PM <: HarmonicArray}
   period = psys.period
   p, m = size(psys); pm = p*m
   ahr = Array{Complex{T},3}(undef, m, p, nh+1)
   copyto!(view(ahr,1:m,1:p,1),complex.(reshape(view(x,1:pm),p,m)))
   for i in 1:nh
      copyto!(view(ahr,1:m,1:p,i+1),complex.(reshape(view(x,i*pm+1:(i+1)*pm),m,p),reshape(view(x,(i+1)*pm+1:(i+2)*pm),m,p)))
   end
   KK = HarmonicArray{:c,T}(ahr, period, 1)
   return maximum(real(psceig(psys.A+psys.B*KK*psys.C,K))) 
end
"""
    pdpofstab_sw(psys, ns = missing; vinit, optimizer, maxit, vtol, Jtol, gtol, show_trace) -> (Fstab,info)

For a discrete-time periodic system `psys = (A(t), B(t), C(t), D(t))` determine a periodic output feedback gain matrix 
`Fstab(t)` of the same period,  
such that the characteristic exponents `Λ` of the closed-loop state-matrix `A(t)+B(t)*Fstab(t)*inv(I-D(t)*Fstab(t))*C(t)` are stable. 
The matrices of the system `psys` are of type `PeriodicArray`. 
The switching times for the resulting switching periodic gain `Fstab(t)` are specified by the 
`N`-dimensional integer vector `ns`. 
By default, `ns = [N]`, where `N` is the maximal number of samples (i.e., `N = psys.period/psys.Ts`), which corresponds to a constant gain.  

The output feedback gain `Fstab(t)` is computed as `Fstab(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)` 
defined as 

    F(t) = F_i for t ∈ [ns[i]Δ,ns[i+1]Δ) and i ∈ {1, ..., N-1}, or
    F(t) = F_N for t ∈ [ns[N]Δ,T),
         
where `T` is the system period (i.e., `T = psys.period`), `Δ` is the system sampling time (i.e., `Δ = psys.Ts`),  
and `F_i` is the `i`-th gain. 
The resulting periodic matrix `Fstab(t)` is of type `SwitchingPeriodicArray`.
The corresponding closed-loop periodic system can be obtained using the function [`psfeedback`](@ref).

For the determination of the optimal feedback gains `F_i` for `i = 1, ...., N` 
an optimization-based approach is employed using using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-free _Nelder-Mead_ local search method for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = Optim.NelderMead()`.   
The alternative gradient-free _Simulated Annealing_ global search method can be selected with
`optimizer = Optim.SimulatedAnnealing()`. 

For a system with `m` inputs and `p` outputs, 
an internal optimization variable `v` is used, defined as an `m×p×N` array. 
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m×p×N` array `v0`.  

The performance index to be minimized is `J := sdeg(v)`, 
where `sdeg(v)` is the stability degree defined as the largest modulus of the characteristic exponents 
of `Af(t) := A(t)+B(t)*F(t)*C(t)`. The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` is the method specific main convergence tolerance (default: `gtol = 1e-3`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
(see the documentation of the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package for additional information). 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 
"""
function pdpofstab_sw(psys::PeriodicStateSpace{PM}, ns::Union{AbstractVector{<:Int},Missing} = missing; vinit::Union{AbstractArray{<:Real,3},Missing} = missing, 
   optimizer = NelderMead(), maxit = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show_trace = false) where {PM <: PeriodicArray}
   p, m = size(psys)
   T = eltype(psys.A)

   if ismissing(ns)
      #N = psys.A.dperiod*psys.A.nperiod
      #ns = collect(1:N)
      ns = [psys.A.dperiod*psys.A.nperiod]
      N = 1
   else
      N = length(ns)
      ns[1] > 0 || error("ns must have only strictly increasing positive values")
      for i in 1:N-1
          ns[i+1] > ns[i] || error("ns must have only strictly increasing positive values")
      end
   end
   if ismissing(vinit) 
      x = zeros(m,p,N)
   else
      (m,p,N) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($m,$p,$N)")) 
      x = copy(vinit)
   end
   smarg = 1-sqrt(eps(T))
   sdeg0 = pssdeg_sw(x, psys, ns)
   show_trace && println("initial stability degree = $sdeg0")

   result = optimize(x->pssdeg_sw(x,psys,ns), x, optimizer, 
      Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   xopt = result.minimizer
   fopt = minimum(result)
   fopt >= sdeg0  && (@warn "no improvement of stability degree achieved")

   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = fopt, optres = result)
   Fopt = Kbuild_sw(xopt,psys.D,ns)
   fopt < smarg || @warn "no stabilization achieved: increase the number of time values"

   return Fopt, info
end
function pssdeg_sw(x::AbstractArray{<:Real,3},psys::PeriodicStateSpace{PM}, ns::AbstractVector{<:Int}) where {PM <: PeriodicArray}
   F = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, psys.period))
   temp = psys.A+psys.B*F*psys.C
   return maximum(abs.(psceig(temp)))
end
"""
    pdpofstab_hr(psys, nh = 0; vinit, optimizer, maxit, vtol, Jtol, gtol, show_trace) -> (Fstab,info)

For a discrete-time periodic system `psys = (A(t), B(t), C(t), D(t))` of period `T` determine a periodic output feedback gain matrix 
`Fstab(t)` of the same period,  
such that the characteristic exponents `Λ` of the closed-loop state-matrix `A(t)+B(t)*Fstab(t)*inv(I-D(t)*Fstab(t))*C(t)` are stable. 
The matrices of the system `psys` are of type `HarmonicArray`. 

The resulting output feedback gain `Fstab(t)` is of type `PeriodicMatrix` and is
computed as `Fstab(t) = inv(I+F(t)D(t))*F(t)`, with `F(t)`, of type `PeriodicMatrix`,
built by sampling, with the sample time `Δ = abs(psys.Ts)`, the harmonic representation form 

                  nh
    Fh(t) = F0 +  ∑ ( Fc_i*cos(i*t*2*π/T)+Fs_i*sin(i*2*π*t/T) ) ,
                 i=1 

where `F0` is the constant term, `Fc_i` is the `i`-th cosinus coefficient matrix and `Fs_i` is the `i`-th sinus coefficient matrix. 
`F(t)` is defined as  `F(t) = Fh((Δ(i-1))`)' for t ∈ [Δ(i-1), Δi), i = 1, ..., T/Δ. 
By default, the number of harmonics is `nh = 0` (i.e., constant output feedback is used). 
The corresponding closed-loop periodic system can be obtained using the function [`psfeedback`](@ref).

For the determination of the optimal feedback gains `F0`, `Fc_i` and `Fs_i` for `i = 1, ...., nh` 
an optimization-based approach is employed using using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-free _Nelder-Mead_ local search method for unconstrained minimizations 
is employed using the keyword argument setting `optimizer = Optim.NelderMead()`.   
The alternative gradient-free _Simulated Annealing_ global search method can be selected with 
`optimizer = Optim.SimulatedAnnealing()`. 

For a system with `m` inputs and `p` outputs, 
an internal optimization variable `v` is used, formed as an `m*p*(2*nh+1)` dimensional vector 
`v := [vec(F0); vec(Fc_1); vec(Fs_1), ... ; vec(Fc_nh); vec(Fs_nh)]'. 
The performance index to be minimized is `J := sdeg(v)`, 
where `sdeg(v)` is the stability degree defined as the largest modulus of the 
characteristic exponents 
of `Af(t) := A(t)+B(t)*F(t)*C(t)`. 
By default, `v` is initialized as `v = 0` (i.e., a zero array of appropriate dimensions). 
The keyword argument `vinit = v0` can be used to initialize `v` with an arbitrary `m*p*(2*nh+1)` array `v0`.  

The optimization process is controlled using several keyword parameters. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterion `J` (default: `Jtol = 0`), 
while `gtol` is the method specific main convergence tolerance (default: `gtol = 1e-3`). 
With the keyword argument `show_trace = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show_trace = false`).   
(see the documentation of the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package for additional information). 

The returned named tuple `info` contains `(fopt, sdeg0, sdeg, vopt, optres)`, where:

`info.fopt` is the resulting value of the optimal performance `J`;

`info.sdeg0` is the initial stability degree of the closed-loop characteristic exponents;

`info.sdeg` is the resulting stability degree of the closed-loop characteristic exponents;

`info.vopt` is the resulting value of the optimization variable `v`; 

`info.optres` is the result returned by the `Optim.optimize(...)` function of the 
[`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) package; 
several functions provided by this package
can be used to inquire various information related to the optimization results
(see the documention of this package). 
"""
function pdpofstab_hr(psys::PeriodicStateSpace{PM}, nh::Int = 0; vinit::Union{AbstractVector{<:Real},Missing} = missing, 
                      optimizer = NelderMead(), maxit = 1000, vtol = 0., Jtol = 0., gtol = 1e-3, show_trace = false) where {PM <: PeriodicMatrix}
   period = psys.period
   p, m = size(psys)

   A = psys.A; B = psys.B; C = psys.C; 
   T = eltype(A)
 
   lx = m*p*(2nh+1)
   if ismissing(vinit) 
      x = zeros(lx)
   else
      length(vinit) == lx || throw(ArgumentError("length of vinit must be $lx")) 
      x = copy(vinit)
   end


   smarg = 1-sqrt(eps()); 
   sdeg0 = pssdeg_hr(x,psys,nh)
   show_trace && println("initial stability degree = $sdeg0")

   result = optimize(x->pssdeg_hr(x,psys,nh), x, optimizer, 
       Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show_trace))
   xopt = result.minimizer
   fopt = minimum(result)
   fopt >= sdeg0  && (@warn "no improvement of stability degree achieved")
      
   info = (vopt = xopt, fopt = fopt, sdeg0 = sdeg0, sdeg = fopt, optres = result)
   Fopt = Fbuild_hr(xopt,psys,nh)
   fopt < smarg || @warn "no stabilization achieved: increase the number of time values"
   return Fopt, info

end
function pssdeg_hr(x::AbstractVector{T}, psys::PeriodicStateSpace{PM}, nh::Int) where {T <: Real, PM <: PeriodicMatrix}
   period = psys.period
   p, m = size(psys); pm = p*m
   Ts = abs(psys.Ts)
   ns = Int(round(period/Ts))
   fhr = Array{Complex{T},3}(undef, m, p, nh+1)
   copyto!(view(fhr,1:m,1:p,1),complex.(reshape(view(x,1:pm),p,m)))
   for i in 1:nh
      copyto!(view(fhr,1:m,1:p,i+1),complex.(reshape(view(x,i*pm+1:(i+1)*pm),m,p),reshape(view(x,(i+1)*pm+1:(i+2)*pm),m,p)))
   end
   Fhr = HarmonicArray{:c,T}(fhr, period, 1)
   Fp = PeriodicMatrix([tpmeval(Fhr,(i-1)*Ts) for i in 1:ns],period)
   ptemp = psys.A+psys.B*Fp*psys.C
   return maximum(abs.(psceig(ptemp)))
end
function Fbuild_hr(x::AbstractVector{T}, psys::PeriodicStateSpace{PM}, nh::Int) where {T <: Real, PM <: PeriodicMatrix}
   period = psys.period
   p, m = size(psys); pm = p*m
   Ts = abs(psys.Ts)
   ns = Int(round(period/Ts))
   fhr = Array{Complex{T},3}(undef, m, p, nh+1)
   copyto!(view(fhr,1:m,1:p,1:1),complex.(reshape(view(x,1:pm),p,m)))
   [copyto!(view(fhr,1:m,1:p,i+1), complex.(view(x,i*pm+1:(i+1)*pm),view(x,(i+1)*pm+1:(i+2)*pm))) for i in 1:nh]
   Fhr = HarmonicArray{:c,T}(fhr, period, 1)
   Fp = PeriodicMatrix([tpmeval(Fhr,(i-1)*Ts) for i in 1:ns],period)

   if iszero(psys.D)
      return Fp
   else
      temp = inv(I+Fp*psys.D)*Fp
      norm(Fp) < sqrt(eps(T))*norm(temp) && (@warn "possible unbounded feedback gain")
      return temp
   end
end

