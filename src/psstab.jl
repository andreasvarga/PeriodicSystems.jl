
"""
    plqofc(psys, Q, R, S; G = I, sdeg = 1, optimizer = Optim.LBFGS(), vinit, maxiter, vtol, Jtol, gtol, show) -> (K, info)

Compute for the discrete-time periodic state-space system `psys = (A(t),B(t),C(t),D(t))`,  
the optimal periodic feedback gain `K(t)` in the output 
feedback control law  `u(t) = K(t)*y(t)`, which minimizes the expectation of the quadratic index: 

     J = E{ Sum [x(t)'*Q(t)*x(t) + 2*x(t)'*S(t)*u(t) + u(t)'*R(t)*u(t)] },

where `Q(t)`, `R(t)`, `S(t)` are periodic weighting matrices. 
By default `S(t)` is missing, in which case, `S(t) = 0` is assumed.   
The covariance of the initial state `x(0)` can be specified via the keyword argument `G` (default: `G = I`)
and a desired stability degree of the closed-loop characteristic multipliers can be specified using
the keyword argument `sdeg` (default: `sdeg = 1`). 

For the determination of the optimal feedback gain an optimization-based approach is employed using 
tools available in the optimization package [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 
By default, the gradient-based limited-memory quasi-Newton method (also known as `L-BFGS`) for unconstrained minimizations 
is employed, using the formulas for function and gradient evaluations developed in [1]. 
Alternative gradient-based methods can be selected using the keyword argument `optimizer`, 
such as, for example,  the quasi-Newton method `BFGS` with `optimizer = Optim.BFGS()`, or 
for small size optimization problems, the Nelder-Mead gradient-free method with `optimizer = Optim.NelderMead()`. 
The keyword parameter `maxiter = maxit` can be used to specify the maximum number of iterations to be performed (default: `maxit = 1000`).
The keyword argument `vtol` can be used to specify the absolute tolerance in 
the changes of the optimization variable `v` (default: `vtol = 0`). The keyword argument `Jtol` can be used to specify the
relative tolerance in the changes of the optimization criterium `J` (default: `Jtol = 0`), 
while `gtol` can be used to specify the absolute tolerance in the gradient  `∇J`, in infinity norm (default: `gtol = 1e-5`). 
With the keyword argument `show = true`,  a trace of the optimization algorithm's state is shown on `stdout` (default `show = false`).    

A bound-constrained optimization can be performed using the keyword argument
`lub = (lb, ub)`, where `lb` is the lower bound and `ub` is the upper bound on the feedback gains. 
By default,  `lub = missing` in which case the bounds `lb = -Inf` and `ub = Inf` are assumed. 

An initial value for the optimization variable `v` can be provided using the keyword argument `vinit = gains`, with `gains`
a `m×p×N` array, where `m` and `p` are the numbers of system inputs and outputs, respectively, and `N` is the number of 
samples (i.e., `N = psys.period/psys.Ts`). By default, `vinit = missing`, in which case a zero matrix `gains = 0` is assumed.  

_References_     

[1] A. Varga and S. Pieters. Gradient-based approach to solve optimal periodic output feedback problems. 
    Automatica, vol.34, pp. 477-481, 1998.
"""
function plqofc(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
                S::Union{AbstractMatrix,PM3,Missing} = missing; kwargs...) where 
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
         #size(S,1) == size(psys.A,1)  || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = convert(PeriodicArray,S) 
      else
         m == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
         nmax == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = S
      end
   end

   _, info = plqofc(convert(PeriodicStateSpace{PeriodicArray},psys), Qa, Ra, Sa; kwargs...)
   return K_build(info.xopt,psys.D), info
end


function plqofc(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
   S::Union{AbstractMatrix,PM3,Missing} = missing; sdeg = 1, G = I, lub = missing,
   vinit::Union{AbstractArray{<:Real,3},Missing} = missing, optimizer = Optim.LBFGS(), maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show = false) where 
   {PM <: PeriodicArray, PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <:PeriodicArray}
   n = size(psys.A.M,2)
   p, m = size(psys); 
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
   m == size(R,2) || throw(ArgumentError("R and B have incompatible dimensions"))
   if !ismissing(S)
      m == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
      n == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
   end

   sdeg <= 1 || throw(ArgumentError("desired stability degree must not exceed 1"))
   N = psys.A.dperiod*psys.A.nperiod

   ismissing(vinit) || (m,p,N) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($p,$m,$N)")) 
   ismissing(vinit) ? x = zeros(m,p,N) : x = copy(vinit)
   A = psys.A; B = psys.B; C = psys.C; 
   K0 = PeriodicArray(x, A.period)
   stlim = 1-sqrt(eps()); 
   sdeg0 = maximum(abs.(pseig(A+B*K0*C)))
   if sdeg < 1 
      scal = 1/(sdeg^(1/N)); A = psys.A*scal; B = psys.B*scal
   end 
   T = eltype(A)
   Gt = similar(Array{T,3},n,n,N)
   Gt[:,:,N] = G == I ? Matrix{T}(I(n)) : G
   for i = 1:N-1
       Gt[:,:,i] = zeros(T,n,n)
   end
   GR = PeriodicArray(Gt, A.period)
   ismissing(lub) || (lb = fill(T(lub[1]),m,p,N); ub = fill(T(lub[2]),m,p,N))

   nG =  (optimizer == NelderMead())

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   WORK = (similar(Matrix{T},m,n), similar(Matrix{T},m,n), similar(Matrix{T},n,p))
   # WORK1 = (X, Y, At, xt, Q, pschur_ws)
   WORK1 = (Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(n, N) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   qr_ws = QRWs(zeros(8), zeros(4))
   WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
            Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
            qr_ws, QROrmWs(zeros(4), qr_ws.τ))


   evs = 2; nit = 100; it = 1; sd = 0; result = []
   while it <= nit && evs >= 1 
      # determine actual stability degree; 
      # scale/shift, if necessary, to make A stable
      KK = PeriodicArray(x, A.period)
      sd = maximum(abs.(pseig(A+B*KK*C)))
      if sd >= stlim
         evs = 1.01*sd;
         scal = 1/(evs^(1/N)); 
         As = scal*A; Bs = scal*B; 
         maxit = 20
      else
         evs = sd; 
         As = A; Bs = B
         maxit = maxiter
      end
      show && println("Current stability degree: $sd")
         
      # solve an unconstrained or bound constrained minimization problem by employing 
      # function and gradient values
      par = (As, Bs, C, R, Q, S, GR, WORK, WORK1, WORK2)
      if ismissing(lub)
         result = optimize(Optim.only_fg!((F,G,x) -> plqofcfungrad!(F, G, x, par)), x, optimizer,
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show))
      else
         result = optimize(Optim.only_fg!((F,G,x) -> plqofcfungrad!(F, G, x, par)), lb, ub, x, Fminbox(optimizer),
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show))
      end
                        
      x = result.minimizer
      it = it+1;
   end
   KK = PeriodicArray(x, A.period)
   sd = maximum(abs.(pseig(psys.A+psys.B*KK*C)))
   info = (xopt = x, sdeg0 = sdeg0, sdeg = sd, fopt = minimum(result), result = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"
   Kopt = K_build(x,psys.D)

   return Kopt, info

end

function plqofcfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for plqofc
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

   (X, Y, At, xt, Q, pschur_ws) = WORK1
   if isnothing(Grad)
      try
         copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
         pslyapd!(X, At, QR.M, xt, Q; adj = true, stability_check = true)
         return tr(X[:,:,1]*GR.M[:,:,end])
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   try
      copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
      pslyapd2!(Y, X, At, GR.M, QR.M, xt, Q, WORK2, pschur_ws; stability_check = true) 
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
function K_build(x::Array{T,3}, D::PM) where {T, PM <: PeriodicMatrix}
   K = PeriodicMatrix([x[:,:,i] for i in 1:size(x,3)], D.period) 
   return iszero(D) ? K : inv(I+K*D)*K
end
function K_build(x::Array{T,3}, D::PM) where {T, PM <: PeriodicArray}
   K = PeriodicArray(x, D.period) 
   return iszero(D) ? K : inv(I+K*D)*K
end
"""
    plqofc_sw(psys, Q, R, S; ns, vinit, kwargs...) -> (K, info)

Compute for the discrete-time periodic state-space system `psys = (A(t),B(t),C(t),D(t))`,  
the optimal switching periodic feedback gain `K(t)` in the output 
feedback control law  `u(t) = K(t)*y(t)`, which minimizes the expectation of the quadratic index: 

     J = E{ Sum [x(t)'*Q(t)*x(t) + 2*x(t)'*S(t)*u(t) + u(t)'*R(t)*u(t)] },

where `Q(t)`, `R(t)`, `S(t)` are periodic weighting matrices. 
The switching times are specified by the `N`-dimensional integer vector `ns` and the resulting switching 
periodic gain `K(t)` is defined by `N` values (see [`SwitchingPeriodicArray`](@ref)). 
By default, `ns = 1:N`, where `N` is the maximal number of samples (i.e., `N = psys.period/psys.Ts`). 

If an initial value for the optimization variable `v` is provided using the keyword argument `vinit = gains`, then `gains`
must be an `m×p×N` array, where `m` and `p` are the numbers of system inputs and outputs.
By default, `vinit = missing`, in which case a zero matrix `gains = 0` is assumed.  
   
The rest of keyword parameters contained in `kwargs` are the same as those used for the function [`plqofc`](@ref). 

The computation of gradient fully exploits the switching structure of the feedback gain, using formulas which generalize
the case of constant feedback considered in [1].     

_References_     

[1] A. Varga and S. Pieters. Gradient-based approach to solve optimal periodic output feedback problems. 
    Automatica, vol.34, pp. 477-481, 1998.
"""
function plqofc_sw(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
   S::Union{AbstractMatrix,PM3,Missing} = missing; sdeg = 1, G = I, lub = missing, ns = missing, 
   vinit::Union{AbstractArray{<:Real,3},Missing} = missing, optimizer = Optim.LBFGS(), maxiter = 1000, vtol = 0., Jtol = 0., gtol = 1e-5, show = false) where 
   {PM <: PeriodicArray, PM1 <: PeriodicArray, PM2 <: PeriodicArray, PM3 <:PeriodicArray}
   n = size(psys.A.M,2)
   p, m = size(psys); 
   issymmetric(Q) || throw(ArgumentError("Q must be a symmetric periodic array"))
   n == size(Q,2) || throw(ArgumentError("Q and A have incompatible dimensions"))
   issymmetric(R) || throw(ArgumentError("R must be a symmetric periodic matrix"))
   m == size(R,2) || throw(ArgumentError("R and B have incompatible dimensions"))
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
      x = zeros(m,p,ps)
   else
      (m,p,ps) == size(vinit) || throw(ArgumentError(" dimensions of vinit must be ($p,$m,$ps)")) 
      x = copy(vinit)
   end

   A = psys.A; B = psys.B; C = psys.C; 
   #Ksw = SwitchingPeriodicArray(x, ns, A.period)
   K0 = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, A.period))
   stlim = 1-sqrt(eps()); 
   sdeg0 = maximum(abs.(pseig(A+B*K0*C)))
   if sdeg < 1 
      scal = 1/(sdeg^(1/N)); A = psys.A*scal; B = psys.B*scal
   end 
   T = eltype(A)
   Gt = similar(Array{T,3},n,n,N)
   Gt[:,:,N] = G == I ? Matrix{T}(I(n)) : G
   for i = 1:N-1
       Gt[:,:,i] = zeros(T,n,n)
   end
   GR = PeriodicArray(Gt, A.period)
   ismissing(lub) || (lb = fill(T(lub[1]),m,p,N); ub = fill(T(lub[2]),m,p,N))

   nG =  (optimizer == NelderMead())

   # preallocate workspaces
   # WORK = (temp1(m x n), temp2(m x n), temp3(n x p)
   WORK = (similar(Matrix{T},m,n), similar(Matrix{T},m,n), similar(Matrix{T},n,p))
   # WORK1 = (WSGrad, X, Y, At, xt, Q, pschur_ws)
   WORK1 = (Array{T,3}(undef, m, p, N), Array{T,3}(undef, n, n, N), nG ? nothing : Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(n, N) )
   # WORK2 = (G, WUSD, WUD, WUL, WY, W, qr_ws, ormqr_ws)
   qr_ws = QRWs(zeros(8), zeros(4))
   WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
            Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
            qr_ws, QROrmWs(zeros(4), qr_ws.τ))


   evs = 2; nit = 100; it = 1; sd = 0; result = []
   while it <= nit && evs >= 1 
      # determine actual stability degree; 
      # scale/shift, if necessary, to make A stable
      KK = convert(PeriodicArray,SwitchingPeriodicArray(x, ns, A.period))
      sd = maximum(abs.(pseig(A+B*KK*C)))
      if sd >= stlim
         evs = 1.01*sd;
         scal = 1/(evs^(1/N)); 
         As = scal*A; Bs = scal*B; 
         maxit = 20
      else
         evs = sd; 
         As = A; Bs = B
         maxit = maxiter
      end
      show && println("Current stability degree: $sd")
         
      # solve an unconstrained or bound constrained minimization problem by employing 
      # function and gradient values
      par = (As, Bs, C, R, Q, S, ns, GR, WORK, WORK1, WORK2)
      if ismissing(lub)
         result = optimize(Optim.only_fg!((F,G,x) -> plqofcswfungrad!(F, G, x, par)), x, optimizer,
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show))
      else
         result = optimize(Optim.only_fg!((F,G,x) -> plqofcswfungrad!(F, G, x, par)), lb, ub, x, Fminbox(optimizer),
                        Optim.Options(x_tol = vtol, f_tol = Jtol, g_tol = gtol, iterations = maxit, show_trace=show))
      end
                        
      x = result.minimizer
      it = it+1;
   end
   KK = SwitchingPeriodicArray(x, ns, A.period)
   sd = maximum(abs.(pseig(psys.A+psys.B*convert(PeriodicArray,KK)*C)))
   info = (xopt = x, sdeg0 = sdeg0, sdeg = sd, fopt = minimum(result), result = result)
   sd <= sdeg || @warn "achieved stability degree $sd larger than desired one $sdeg"
   Kopt = K_build(x,psys.D,ns)

   return Kopt, info

end
function plqofcswfungrad!(Fun,Grad,x,par) 
   # generic function/gradient evaluation for plqofc
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

   (WSGrad, X, Y, At, xt, Q, pschur_ws) = WORK1
   if isnothing(Grad)
      try
         copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
         pslyapd!(X, At, QR.M, xt, Q; adj = true, stability_check = true)
         return tr(X[:,:,1]*GR.M[:,:,end])
      catch te
         isnothing(findfirst("stability",te.msg)) && (@show te; error("unknown error"))
         return 1.e20
      end
   end
   try
      copyto!(view(At,:,:,:),view(Ar.M,:,:,:))
      pslyapd2!(Y, X, At, GR.M, QR.M, xt, Q, WORK2, pschur_ws; stability_check = true) 
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
      # @show ns
      # @show WSGrad-Grad
      # error("stop")
   end

   isnothing(Fun) ? (return nothing) : (return tr(X[:,:,1]*GR.M[:,:,end]))
end

function K_build(x::Array{T,3}, D::PM, ns) where {T, PM <: PeriodicArray}
   K = SwitchingPeriodicArray(x, ns, D.period) 
   return iszero(D) ? K : inv(I+K*convert(SwitchingPeriodicArray,D))*K
end
function plqofc_sw(psys::PeriodicStateSpace{PM}, Q::Union{AbstractMatrix,PM1}, R::Union{AbstractMatrix,PM2}, 
                S::Union{AbstractMatrix,PM3,Missing} = missing; ns = missing, kwargs...) where 
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
         #size(S,1) == size(psys.A,1)  || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = convert(PeriodicArray,S) 
      else
         m == size(S,2) || throw(ArgumentError("S and B have incompatible dimensions"))
         nmax == size(S,1) || throw(ArgumentError("S and A have incompatible dimensions"))
         Sa = S
      end
   end
   N = psys.A.dperiod*psys.A.nperiod
   ismissing(ns) && (ns = collect(1:N))

   _, info = plqofc_sw(convert(PeriodicStateSpace{PeriodicArray},psys), Qa, Ra, Sa; ns, kwargs...)
   return K_build(info.xopt,psys.D,ns), info
end
function K_build(x::Array{T,3}, D::PM, ns) where {T, PM <: PeriodicMatrix}
   K = SwitchingPeriodicMatrix([x[:,:,i] for i in 1:length(ns)], ns, D.period) 
   return iszero(D) ? K : inv(I+K*convert(SwitchingPeriodicMatrix,D))*K
end



