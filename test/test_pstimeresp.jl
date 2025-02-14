module Test_pstimeresp

using PeriodicSystems
using ApproxFun
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra


println("Test_pstimeresp")



@testset "pstimeresp" begin


# step response standard system
a = [0 1; -2 -4]; b = [2 1;0 1]; c = [0.5 1]; d = [0 1]; x0 = [1,2]; u0 = [1, 1]; Ts = 1; timesteps = 20

sysc = dss(a,b,c,d);
@time sysd, xd0, = c2d(sysc, Ts; x0, u0); 
u = ones(timesteps+1,2);
@time y, tout, x = timeresp(sysd, u, Int[], xd0; state_history = true);

@time y1, tout1, x1 = timeresp(sysc, u, tout, x0; state_history = true, interpolation = "zoh")
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7

@time ys, touts, xs = stepresp(sysd, tout[end]; x0, timesteps, state_history = true);

@time ys1, touts1, xs1 = stepresp(sysc, tout[end]; x0, timesteps, state_history = true)
@test norm(ys-ys1,Inf) < 1.e-7 && norm(xs-xs1,Inf) < 1.e-7

psysc = ps(dss(a,b,c,d),10);
@time psysd = psc2d(psysc, Ts); 
@time yp, toutp, xp = pstimeresp(psysd, u, Int[], x0; state_history = true);
@test norm(yp-y1,Inf) < 1.e-7 && norm(xp-x1,Inf) < 1.e-7

@time yp1, toutp1, xp1 = pstimeresp(psysc, u, toutp, x0; state_history = true);
@test norm(yp-yp1,Inf) < 1.e-7 && norm(xp-xp1,Inf) < 1.e-7

@time yps, toutps, xps = psstepresp(psysd, toutp[end]; x0, timesteps, state_history = true)

@time yps1, toutps1, xps1 = psstepresp(psysc, toutp[end]; x0, timesteps, state_history = true)
@test norm(yps-yps1,Inf) < 1.e-7 && norm(xps-xps1,Inf) < 1.e-7

@time yps, toutps, xps = psstepresp(psysc,15; x0, timesteps = 150, state_history = true); 
@time yps1, toutps1, xps1 = psstepresp(psysc,10; x0, timesteps = 100,state_history = true); 
@test norm(yps1[:,:,:]-yps[1:101,:,:],Inf) < 1.e-7 && norm(xps1[:,:,:]-xps[1:101,:,:],Inf) < 1.e-7


@time yps, toutps, xps = psstepresp(psysc,15; x0, timesteps = 100, state_history = true); 

PM = PeriodicTimeSeriesMatrix
for PM in (HarmonicArray, FourierFunctionMatrix, PeriodicSymbolicMatrix, PeriodicTimeSeriesMatrix)
   psysc1 = convert(PeriodicStateSpace{PM},psysc);
   @time yps1, toutps1, xps1 = psstepresp(psysc1,15; x0, timesteps = 100, state_history = true); 
   @test norm(yps1[:,:,:]-yps,Inf) < 1.e-7 && norm(xps1[:,:,:]-xps,Inf) < 1.e-7
end


# response of standard system to stairs input
u = rand(11,2);
sysc = dss(a,b,c,d);
@time sysd, xd0, = c2d(sysc, Ts; x0, u0); 
@time y, tout, x = timeresp(sysd, u, Int[], xd0; state_history = true);

@time y1, tout1, x1 = timeresp(sysc, u, tout, x0; state_history = true, interpolation = "zoh")
@test norm(y-y1,Inf) < 1.e-7 && norm(x-x1,Inf) < 1.e-7


psysc = ps(dss(a,b,c,d),10);
@time psysd = psc2d(psysc, Ts); 
@time yp, toutp, xp = pstimeresp(psysd, u, Int[], x0; state_history = true);
@test norm(yp-y1,Inf) < 1.e-7 && norm(xp-x1,Inf) < 1.e-7

@time yp1, toutp1, xp1 = pstimeresp(psysc, u, toutp, x0; state_history = true);
@test norm(yp-yp1,Inf) < 1.e-7 && norm(xp-xp1,Inf) < 1.e-7

# periodic array
n = 5; nu = 2; ny = 3; pa = 3; pb = 6; pc = 2; pd = 1;   
Ad = 0.1*PeriodicArray(rand(Float64,n,n,pa),pa);
Bd = PeriodicArray(rand(Float64,n,nu,pb),pb);
Cd = PeriodicArray(rand(Float64,ny,n,pc),pc);
Dd = PeriodicArray(rand(Float64,ny,nu,pd),pd);
psysd = ps(Ad,Bd,Cd,Dd);

u = rand(11,nu); x0 = rand(n); 
@time yp, toutp, xp = pstimeresp(psysd, u, Int[], x0; state_history = true);

@time yps, toutps, xps = psstepresp(psysd, toutp[end]; x0, timesteps, state_history = true)

# periodic matrix
psysd1 = convert(PeriodicStateSpace{PeriodicMatrix},psysd);
@time yp1, toutp1, xp1 = pstimeresp(psysd1, u, toutp, x0; state_history = true);
@test norm(yp-yp1,Inf) < 1.e-7 && norm(xp-xp1,Inf) < 1.e-7

@time yps1, toutps1, xps1 = psstepresp(psysd1, toutp[end]; x0, timesteps, state_history = true)
@test norm(yps-yps1,Inf) < 1.e-7 && norm(xps-xps1,Inf) < 1.e-7


# periodic matrix time-varying dimensions
p = 5; na = [10, 8, 6, 4, 2]; ma = circshift(na,-1); nu = 2; ny = 3; 
period = 10;
Ad = 0.001*PeriodicMatrix([rand(ma[i],na[i]) for i in 1:p],period);
Bd = PeriodicMatrix([rand(ma[i],nu) for i in 1:p],period);
Cd = PeriodicMatrix([rand(ny,na[i]) for i in 1:p],period);
Dd = PeriodicMatrix(rand(ny,nu),period; nperiod = rationalize(Ad.period/Ad.Ts).num);
psysd = ps(Ad,Bd,Cd,Dd);

u = rand(11,nu); x0 = rand(na[1]); 
@time yp, toutp, xp = pstimeresp(psysd, u, Int[], x0; state_history = true);
@time yps, toutps, xps = psstepresp(psysd, toutp[end]; x0, timesteps, state_history = true)

psysd1 = convert(PeriodicStateSpace{PeriodicArray},psysd);
@time yp1, toutp1, xp1 = pstimeresp(psysd1, u, toutp, x0; state_history = true);
@test norm(yp-yp1,Inf) < 1.e-7 && norm(xp-xp1,Inf) < 1.e-7

@time yps1, toutps1, xps1 = psstepresp(psysd1, toutp[end]; x0, timesteps, state_history = true)
@test norm(yps-yps1,Inf) < 1.e-7 && norm(xps-xps1,Inf) < 1.e-7

# Pitelkau's example 
ω = 0.00103448
period = 2*pi/ω
a = [  0            0     5.318064566757217e-02                         0
       0            0                         0      5.318064566757217e-02
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
c = [1 0 0 0;0 1 0 0];
d = zeros(2,1);
psysc = ps(a,PeriodicFunctionMatrix(b,period),c,d);

Ts = period/100; 
@time psysd = psc2d(psysc, Ts, reltol = 1.e-10, abstol = 1.e-10); 
x0 = zeros(4); u = 1.e5*randn(101,1); 
@time yp, toutp, xp = pstimeresp(psysd, u, Int[], x0; state_history = true);

@time yp1, toutp1, xp1 = pstimeresp(psysc, u, toutp, x0; state_history = true, reltol = 1.e-8, abstol = 1.e-8);
@test norm(yp-yp1,Inf) < 1.e-7 && norm(xp-xp1,Inf) < 1.e-7

u = 1.e5*ones(101,1);
@time yp1, toutp1, xp1 = pstimeresp(psysc, u, toutp, x0; state_history = true, reltol = 1.e-8, abstol = 1.e-8);

ustep = 1.e5*ones(1);
@time yps1, toutps1, xps1 = psstepresp(psysc, toutp[end]; ustep, state_history = true, reltol = 1.e-8, abstol = 1.e-8);
@test norm(yp1-yps1,Inf) < 1.e-5 && norm(xp1-xps1,Inf) < 1.e-5

solver = "non-stiff"
for solver in ("non-stiff", "stiff", "symplectic", "auto")
    println("solver = $solver")
    @time yp2, toutp2, xp2 = pstimeresp(psysc, t-> [1.e5], toutp, x0; state_history = true, solver, reltol = 1.e-8, abstol = 1.e-8);
    @test norm(yp1-yp2,Inf) < 1.e-5 && norm(xp1-xp2,Inf) < 1.e-5
end

end # timeresp    

end
