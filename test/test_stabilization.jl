module Test_stabilization

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using LinearAlgebra: BlasInt
using ApproxFun
using MatrixPencils
using Optim
using LineSearches
using QuadGK


println("Test_stabilization")

@testset "Test_output_feedback_stabilization" begin

# Example 1.2 Bittanti-Colaneri
@variables t
A = [-1+sin(t) 0; 1-cos(t) -3];
B = [-1-cos(t); 2-sin(t)];
C = [0 1];
D = [0;;]
Ap = PeriodicSymbolicMatrix(A,2*pi);
Bp = PeriodicSymbolicMatrix(B,2*pi);
Cp = PeriodicSymbolicMatrix(C,2*pi);
Dp = PeriodicSymbolicMatrix(D,2*pi);
#k = 1; R = PeriodicSymbolicMatrix([−1+sin(t) −k−k*cos(t); 1-cos(t) −3+2k−k*sin(t)],2pi)


psys = convert(PeriodicStateSpace{PeriodicFunctionMatrix},ps(Ap,Bp,Cp,Dp))

@time Kopt, info = pcpofstab_sw(psys)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

@time Kopt, info = pcpofstab_sw(psys, optimizer = SimulatedAnnealing())
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

@time Kopt, info = pcpofstab_sw(psys,gtol = 1.e-5, vinit = info.vopt)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg


@time Kopt, info = pcpofstab_sw(psys,[0.,pi],K=200)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,200))) ≈ info.sdeg


psys = convert(PeriodicStateSpace{HarmonicArray},ps(Ap,Bp,Cp,Dp))
@time Kopt, info = pcpofstab_hr(psys)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

@time Kopt, info = pcpofstab_hr(psys, optimizer = SimulatedAnnealing())
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

@time Kopt, info = pcpofstab_hr(psys,gtol = 1.e-5, vinit = info.vopt)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg


@time Kopt, info = pcpofstab_hr(psys,1,K=200)
psyscl = psfeedback(psys, Kopt; negative=false);
@test maximum(real(psceig(psyscl.A,200))) ≈ info.sdeg
# @test maximum(real(pspole(psyscl,20))) ≈ info.sdeg # this fails to compute correct poles!!


psysd = psc2d(psys,psys.period*0.01)
psysd1 = psc2d(PeriodicMatrix,psys,psys.period*0.01)

@time Kdopt, infod = pdpofstab_sw(psysd; gtol = 1.e-4,optimizer = SimulatedAnnealing())
psyscld = psfeedback(psysd, Kdopt; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infod.sdeg) < 1.e-5

@time Kdopt, infod = pdpofstab_sw(psysd; gtol = 1.e-6)
psyscld = psfeedback(psysd, Kdopt; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infod.sdeg) < 1.e-5

@time Kdopthr, infodhr = pdpofstab_hr(psysd1; gtol = 1.e-6)
psyscld = psfeedback(psysd1, Kdopthr; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infodhr.sdeg) < 1.e-5 && infod.sdeg ≈ infodhr.sdeg

@time Kdopt2, infod2 = pdpofstab_sw(psysd,[50,100]; gtol = 1.e-6)
psyscld = psfeedback(psysd, Kdopt2; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infod2.sdeg) < 1.e-5 && infod2.sdeg < infod.sdeg

@time Kdopthr1, infodhr1 = pdpofstab_hr(psysd1,1; gtol = 1.e-6)
psyscld = psfeedback(psysd1, Kdopthr1; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infodhr1.sdeg) < 1.e-5 && infodhr1.sdeg < infodhr.sdeg


# Belgian chocolate problem  
# Bittanti-Colaneri, p. 35-38 
δ = 0.9
a = [0 1; -1 2δ]; b = [0;1;;]; c = [-2 2δ]; d = [1;;]
sys = dss(a,b,c,d)
#sysd = c2d(sys,0.01)[1]
# sys0 = dss(a,b,c,0*d)


K1 = -4.22; K2 = 2.12; K = PeriodicSwitchingMatrix([[K1],[K2]],[0.,1.],2)
K0 = inv(I+K)*K

psyscl = psfeedback(sys, K0; negative=false)
eig = pspole(psyscl)
@test maximum(real.(eig)) < 0

psys = ps(sys,2)
@time Kopt, info = pcpofstab_sw(psys,[0.,1.]; optimizer = SimulatedAnnealing())
psyscl = psfeedback(psys, Kopt; negative=false)
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

@time Kopt, info = pcpofstab_sw(psys,[0.,1.]; vinit = reshape([K1;K2],1,1,2))
psyscl = psfeedback(psys, Kopt; negative=false)
@test maximum(real(pspole(psyscl,100))) ≈ info.sdeg

sys0 = dss(a,b,c,0*d)
sys01 = dss(a,b,c,0.1*d)
psyshr = ps(HarmonicArray,sys,2)
psyshr0 = ps(HarmonicArray,sys0,2)
psyshr01 = ps(HarmonicArray,sys01,2)

# not stabilizable with constant feedback
@time Fhr, infohr = pcpofstab_hr(psyshr0, K = 100, Jtol = 0.001, gtol = 1.e-4);
@test infohr.sdeg > 0

# stabilization with first order harmonic gain possible if D = 0
@time Fhr, infohr = pcpofstab_hr(psyshr0, 1; K = 100, Jtol = 0.001, gtol = 1.e-4);
psyscl = psfeedback(psyshr0, Fhr; negative=false)
eig = pspole(psyscl,20)
@test maximum(real(eig)) ≈ infohr.sdeg

# stabilization with first order harmonic gain not possible if D = 1, because unbounded gain
@time Fhr, infohr = pcpofstab_hr(psyshr, 1; K = 100, Jtol = 0.001, gtol = 1.e-8)
K = PeriodicSystems.Kbuild_hr(infohr.vopt,psyshr.D,1;PFM = true) # determine gain as a periodic function matrix
@test norm(K(1.3793564585209916)) > 1.e5

# stabilization of a discretized system
sysd = c2d(sys,0.01)[1]
psysd = ps(PeriodicMatrix,sysd,2)
#sysd = rss(3,2,2;disc = true,Ts = 0.01)  # needs DescriptorSystems v1.3.8
@time Kdopt, infod = pdpofstab_hr(psysd,1; gtol = 1.e-5,optimizer = SimulatedAnnealing())
psyscld = psfeedback(psysd, Kdopt; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infod.sdeg) < 1.e-5
@time Kdopt, infod = pdpofstab_hr(psysd,1,vinit = infod.vopt, gtol = 1.e-6)
psyscld = psfeedback(psysd, Kdopt; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infod.sdeg) < 1.e-5

sysd = c2d(sys,0.01)[1]
psysda = ps(PeriodicArray,sysd,2)
#sysd = rss(3,2,2;disc = true,Ts = 0.01)  # needs DescriptorSystems v1.3.8
@time Kdoptsw, infodsw = pdpofstab_sw(psysda,[100,200]; gtol = 1.e-4,optimizer = SimulatedAnnealing())
psyscld = psfeedback(psysda, Kdoptsw; negative=false)
@test abs(maximum(abs.(pspole(psyscld)))-infodsw.sdeg) < 1.e-5

end

@testset "Test_optimal_output_feedback_stabilization (discrete-time)" begin


# Pitelkau's example
ω = 0.00103448
period = 2*pi/ω
a = [  0            0     5.318064566757217e-02                         0
       0            0                         0      5.318064566757217e-02
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
c = [1 0 0 0;0 1 0 0]; d = zeros(2,1); 
#c = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 1]; d = zeros(4,1);
psysc = ps(a,PeriodicFunctionMatrix(b,period),c,d);

K = 120;
#K = 512;
@time psysa = psc2d(psysc,period/K,reltol = 1.e-10);
psys = convert(PeriodicStateSpace{PeriodicMatrix},psysa);

#Q = eye(4); R = eye(1)*1.e-7; 
Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time F, info = pdlqofc(psys, Q, R);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5 
@time F1, info1 = pdlqofc(psys, Q, R,S = zeros(4,1));
@time F2, info2 = pdlqofc(psys, Q, R,S = PeriodicMatrix(zeros(4,1),psys.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt

# perform only stabilization
@time F1, info1 = pdlqofc(psys, Q, R, optimizer = nothing);
psyscl1 = psfeedback(psys, F1; negative=false); 
@test abs(maximum(abs.(pseig(psyscl1.A)))-info1.sdeg) < 1.e-5

Qp = PeriodicMatrix([Q],period); Rp = PeriodicMatrix([R],period); 
Sp = PeriodicMatrix([zeros(4,1)],period);
@time Fp, infop = pdlqofc(psys, Qp, Rp, S = Sp);
@test info.fopt ≈ infop.fopt 

for k in (1, 2, 3, 10, 60, 120)
    @time Fsw, infosw = pdlqofc_sw(psys, Q, R, collect(k:k:120));
    psyscl = psfeedback(psys, Fsw; negative=false); 
    @test abs(maximum(abs.(pseig(psyscl.A)))-infosw.sdeg) < 1.e-5
    println(" k = $k,  sdeg = $(infosw.sdeg), Jopt = $(infosw.fopt)")
    k == 1 && (@test info.fopt ≈ infosw.fopt) 
end

# perform only stabilization
@time F1, info1 = pdlqofc_sw(psys, Q, R, optimizer = nothing);
psyscl1 = psfeedback(psys, F1; negative=false); 
@test abs(maximum(abs.(pseig(psyscl1.A)))-info1.sdeg) < 1.e-5

Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time Fa, info = pdlqofc(psysa, Q, R);
psyscl = psfeedback(psysa, Fa; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5  
@time F1, info1 = pdlqofc(psysa, Q, R, S = zeros(4,1));
@time F2, info2 = pdlqofc(psysa, Q, R,S = PeriodicArray(zeros(4,1),psys.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt
@test norm(convert(PeriodicArray,F)-Fa) < 1.e-7

Qa = PeriodicArray(Q,period); Ra = PeriodicArray(R,period); 
Sa = PeriodicArray(zeros(4,1),period);
@time Fa, infoa = pdlqofc(psysa, Qa, Ra, S = Sa);
@test info.fopt ≈ infoa.fopt 

for k in (1, 2, 3, 10, 60, 120)
    @time Fasw, infosw = pdlqofc_sw(psysa, Q, R, collect(k:k:120));
    psyscl = psfeedback(psysa, Fasw; negative=false); 
    @test abs(maximum(abs.(pseig(psyscl.A)))-infosw.sdeg) < 1.e-5
    println(" k = $k,  sdeg = $(infosw.sdeg), Jopt = $(infosw.fopt)")
    k == 1 && (@test info.fopt ≈ infosw.fopt) 
end


# Pitelkau's example with nonzero D
ω = 0.00103448
period = 2*pi/ω
a = [  0            0     5.318064566757217e-02                         0
       0            0                         0      5.318064566757217e-02
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
c = [1 0 0 0;0 1 0 0]; d = 100*eps()*ones(2,1);  d = 1.e-7*ones(2,1);
#c = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 1]; d = zeros(4,1);
psysc1 = ps(a,PeriodicFunctionMatrix(b,period),c,d);

K = 120;
#K = 512;
@time psysa1 = psc2d(psysc1,period/K,reltol = 1.e-10);
psys1 = convert(PeriodicStateSpace{PeriodicMatrix},psysa1);

#Q = eye(4); R = eye(1)*1.e-7; 
Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time F, info = pdlqofc(psys1, Q, R);
psyscl = psfeedback(psys1, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5 
@time F1, info1 = pdlqofc(psys1, Q, R,S = zeros(4,1));
@time F2, info2 = pdlqofc(psys1, Q, R,S = PeriodicMatrix(zeros(4,1),psys1.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt


@time Fa, infoa = pdlqofc(psysa1, Q, R);
psyscl = psfeedback(psysa1, Fa; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5
@time F1, infoa1 = pdlqofc(psysa1, Q, R,S = zeros(4,1));
@time F2, infoa2 = pdlqofc(psysa1, Q, R, S = PeriodicArray(zeros(4,1),psys1.period));
@test infoa.fopt ≈ infoa1.fopt ≈ infoa2.fopt
@test norm(convert(PeriodicArray,F)-Fa) < 1.e-7

# Pitelkau's example with large sampling time
ω = 0.00103448
period = 2*pi/ω
a = [  0            0     5.318064566757217e-02                         0
       0            0                         0      5.318064566757217e-02
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
c = [1 0 0 0;0 1 0 0]; d = zeros(2,1); 
#c = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 1]; d = zeros(4,1);
psysc = ps(a,PeriodicFunctionMatrix(b,period),c,d);

Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;

K = 20;
psysa = psc2d(psysc,period/K,reltol = 1.e-10);
psys = convert(PeriodicStateSpace{PeriodicMatrix},psysa);
@time F, info = pdlqofc(psys, Q, R, gtol=0.01,lub = (-5000,5000),maxiter=100);
@time Fa, infoa = pdlqofc(psysa, Q, R, gtol=0.01,lub = (-5000,5000),maxiter=100);
@test info.fopt ≈ infoa.fopt 


for K in (10, 20, 40, 120)
    psys = psc2d(psysc,period/K,reltol = 1.e-10);
    psysa = convert(PeriodicStateSpace{PeriodicArray},psys);

    @time F, info = pdlqofc(psys, Q, R, gtol=0.0001);
    @time Fa, infoa = pdlqofc(psysa, Q, R, gtol=0.0001);
    @test info.fopt ≈ infoa.fopt 
    println(" K = $K,  sdeg = $(info.sdeg), fval = $(info.fopt)")
end 

# time-varying dimensions
Ad = PeriodicMatrix([[1. 2], [7;8]],2);
Bd = PeriodicMatrix( [[ 3 ], [ 9; 10]] ,2);
Cd = PeriodicMatrix( [[ 4 5 ], [ 11 ]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1); 
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd);
Q = eye(2); R = eye(1);
@time F, info = pdlqofc(psys, Q, R, gtol=0.01);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5

Q = PeriodicMatrix([eye(2),eye(1)],2); R = PeriodicMatrix([eye(1),2*eye(1)],2); 
S = PeriodicMatrix([0.001*ones(1,1),0.001*ones(2,1)],2);
@time F, info = pdlqofc(psys, Q, R; S, gtol=0.01);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5

@time F, info = pdlqofc(psys, Q, R; S, gtol=0.01, optimizer = NelderMead());
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5

end

@testset "Test_output_feedback_stabilization (continuous-time)" begin

# stabilization with switching feedback

a = t-> [  -1+sin(t) 0; 1-cos(t) -3];
b = t-> [-1-cos(t); 2-sin(t);;]
c = [0 1]; d = [0;;]
period = 2pi
psys = ps(PeriodicFunctionMatrix(a,period),PeriodicFunctionMatrix(b,period),c,d,2pi)
psysu = ps(PeriodicFunctionMatrix(a,period)+1.3I,PeriodicFunctionMatrix(b,period),c,d,2pi)

Q = Matrix{Float64}(I(2)); R = [1.;;]

# optimization with constant feedback
@time Fsw, infosw = pclqofc_sw(psys, Q, R, Jtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad = false); 
@time Fswi, infoswi = pclqofc_sw(psys, Q, R, Jtol = 1.e-5, gtol = 1.e-5, abstol=1.e-7,reltol=1.e-7,quad = true); 
@test maximum(abs.(infoswi.vopt-infosw.vopt)) < 1.e-4 && abs(infoswi.fopt-infosw.fopt) < 1.e-7 

# optimization with switching feedback
@time Fsw2, infosw2 = pclqofc_sw(psys, Q, R, collect(0:9)*pi/5; K = 100, Jtol = 1.e-10, gtol = 1.e-6, abstol=1.e-7,reltol=1.e-7);
@test infosw.sdeg0 > infosw2.sdeg 

# stabilization and optimization with constant feedback
@time Fsw, infosw = pclqofc_sw(psysu, Q, R, Jtol = 1.e-4, abstol=1.e-7,reltol=1.e-7, stabilizer = NelderMead()); 
@test infosw.sdeg0 > 0 && infosw.sdeg < 0

# Belgian chocolate problem  
# Bittanti-Colaneri, p. 35-38 
δ = 0.9
a = [0 1; -1 2δ]; b = [0;1;;]; c = [-2 2δ]; d = [1;;]; 
sys = dss(a,b,c,d)
psys = ps(PeriodicFunctionMatrix,sys,2)

Q = Matrix{Float64}(I(2)); R = [1.;;]

# not stabilizable with constant feedback
try 
  @time Fhsw, infosw = pclqofc_sw(psys, Q, R; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,stabilizer = NelderMead());
catch err
  @test true
  println(err)
end    
# stabilization with switching gain possible 
@time Fsw, infosw = pclqofc_sw(psys, Q, R, [0.,1.]; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,stabilizer = NelderMead());
psyscl = psfeedback(psys, Fsw; negative=false)
eig = pspole(psyscl,100)
@test abs(maximum(real(eig)) - infosw.sdeg) < 1.e-7

# Pitelkau's example

ω0 = 0.00103448 # rad/s
period = 2*pi/ω0
Ix = 5764; Iy = 6147; Iz = 10821; hy = - 420; 
ωn = 5.318064566757217e-02 # wn = -hy/sqrt(Ix*Iz)
a = [  0            0            ωn                  0
       0            0            0                   ωn
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω0*t); -0.3701336*10^-7*cos(ω0*t)];
bw = [0 0; 0 0; 1/(Ix*ωn) 0; 0 1/(Iz*ωn) ]
c = [1 0 0 0;0 1 0 0];
d = zeros(2,1);
dw = zeros(2,2);
psys = ps(a,PeriodicFunctionMatrix(b,period),c,d)
psyss = ps(a-0.01*I,PeriodicFunctionMatrix(b,period),c,d)
psys2 = ps(a,PeriodicFunctionMatrix(b,period),eye(4),zeros(4,1));
Q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; R = [1.e-11;;]; 

# not stabilizable with constant gain
try
   @time Fsw, infosw = pclqofc_sw(psys, Q, R; K = 200, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true);
catch err
   @test true
   println(err)
end    
    
# optimization of a modified stable system 
@time Fsw, infosw = pclqofc_sw(psyss, Q, R, collect(0:9)*psyss.period/10; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=false);
@test maximum(real(psceig(psyss.A+psyss.B*Fsw*psyss.C,100))) ≈ infosw.sdeg && infosw.sdeg < 0

# stabilization and optimization of a system with all state measurable
@time Fsw, infosw = pclqofc_sw(psys2, Q, R, collect(0:199)*psyss.period/200; K = 200, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,N=200);
@test maximum(real(psceig(psys2.A+psys2.B*Fsw*psys2.C,100))) ≈ infosw.sdeg && infosw.sdeg < 0


# stabilization with harmonic feedback

a = t-> [  -1+sin(t) 0; 1-cos(t) -3];
b = t-> [-1-cos(t); 2-sin(t);;]
c = [0 1]; d = [0;;]
period = 2pi

psyshr = ps(convert(HarmonicArray,PeriodicFunctionMatrix(a,period)),convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d,2pi)
psyshru = ps(convert(HarmonicArray,PeriodicFunctionMatrix(a,period)+1.3I),convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d,2pi)

Q = Matrix{Float64}(I(2)); R = [1.;;]

# optimization with constant feedback
@time Fhr, infohr = pclqofc_hr(psyshr, Q, R, Jtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad = false); 
@time Fhri, infohri = pclqofc_hr(psyshr, Q, R, Jtol = 1.e-5, gtol = 1.e-5, abstol=1.e-7,reltol=1.e-7,quad = true); 
@test maximum(abs.(infohri.vopt-infohr.vopt)) < 1.e-4 && abs(infohri.fopt-infohr.fopt) < 1.e-7 

# optimization with second order harmonic feedback
@time Fhr2, infohr2 = pclqofc_hr(psyshr, Q, R, 2; K = 100, Jtol = 1.e-10, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true); 
@test infohr.sdeg0 > infohr2.sdeg 


# stabilization and optimization with constant feedback
@time Fhr, infohr = pclqofc_hr(psyshru, Q, R, Jtol = 1.e-4, abstol=1.e-7,reltol=1.e-7, stabilizer = NelderMead()); 
@test infohr.sdeg0 > 0 && infohr.sdeg < 0


# Belgian chocolate problem  
# Bittanti-Colaneri, p. 35-38 
δ = 0.9
a = [0 1; -1 2δ]; b = [0;1;;]; c = [-2 2δ]; d = [1;;]; 
sys = dss(a,b,c,d)
sys0 = dss(a,b,c,0*d)
sys01 = dss(a,b,c,0.1*d)
psyshr = ps(HarmonicArray,sys,2)
psyshr0 = ps(HarmonicArray,sys0,2)
psyshr01 = ps(HarmonicArray,sys01,2)
Q = Matrix{Float64}(I(2)); R = [1.;;]

# not stabilizable with constant feedback
try 
  @time Fhr, infohr = pclqofc_hr(psyshr0, Q, R, 0; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,stabilizer = NelderMead());
catch err
  @test true
  println(err)
end    
# stabilization with first order harmonic gain possible if D = 0
@time Fhr, infohr = pclqofc_hr(psyshr0, Q, R, 1; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,stabilizer = NelderMead());
psyscl = psfeedback(psyshr0, Fhr; negative=false)
eig = pspole(psyscl,20)
@test maximum(real(eig)) ≈ infohr.sdeg

# stabilization with first order harmonic gain possible if D = 0.1
@time Fhr, infohr = pclqofc_hr(psyshr01, Q, R, 1; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=false,stabilizer = NelderMead());
psyscl = psfeedback(psyshr01, Fhr; negative=false)
eig = pspole(psyscl,20)
@test maximum(real(eig)) ≈ infohr.sdeg

# stabilization with first order harmonic gain not possible if D = 1, because unbounded gain

@time Fhr, infohr = pclqofc_hr(psyshr, Q, R, 1; K = 100, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true,optimizer = NelderMead());
K = PeriodicSystems.Kbuild_hr(infohr.vopt,psyshr.D,1;PFM = true) # determine gain as a periodic function matrix
@test norm(K(0.6139372254026436)) > 1.e5

ω0 = 0.00103448 # rad/s
period = 2*pi/ω0
Ix = 5764; Iy = 6147; Iz = 10821; hy = - 420; 
ωn = 5.318064566757217e-02 # wn = -hy/sqrt(Ix*Iz)
a = [  0            0            ωn                  0
       0            0            0                   ωn
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω0*t); -0.3701336*10^-7*cos(ω0*t)];
bw = [0 0; 0 0; 1/(Ix*ωn) 0; 0 1/(Iz*ωn) ]
c = [1 0 0 0;0 1 0 0];
d = zeros(2,1);
dw = zeros(2,2);
psyshr = ps(a,convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d)
psyshrs = ps(a-0.01*I,convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d)
psyshr2 = ps(a,convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),eye(4),zeros(4,1));
Q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; R = [1.e-11;;]; 

# not stabilizable with 10 harmonics components!!
try
   @time Fhr, infohr = pclqofc_hr(psyshr, Q, R, 10; K = 200, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true);
catch err
   @test true
   println(err)
end    
    
# optization of a modified stable system 
@time Fhr, infohr = pclqofc_hr(psyshrs, Q, R, 10; K = 200, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=false);
@test maximum(real(psceig(psyshrs.A+psyshrs.B*Fhr*psyshrs.C,100))) ≈ infohr.sdeg && infohr.sdeg < 0

# stabilization and optimization of a system with all state measurable
@time Fhr, infohr = pclqofc_hr(psyshr2, Q, R, 2; K = 200, Jtol = 0.001, gtol = 1.e-4, abstol=1.e-7,reltol=1.e-7,quad=true);
@test maximum(real(psceig(psyshr2.A+psyshr2.B*Fhr*psyshr2.C,100))) ≈ infohr.sdeg && infohr.sdeg < 0

end

println("Test_stabilization")

@testset "Test_state_feedback_stabilization" begin

    # Pitelkau's example 
    ω0 = 0.00103448 # rad/s
    period = 2*pi/ω0
    Ix = 5764; Iy = 6147; Iz = 10821; hy = - 420; 
    ωn = 5.318064566757217e-02 # wn = -hy/sqrt(Ix*Iz)
    a = [  0            0            ωn                  0
           0            0            0                   ωn
           -1.352134256362805e-03   0             0    -7.099273035392090e-02
           0    -7.557182037479544e-04     3.781555288420663e-02             0
    ];
    b = t->[0; 0; 0.1389735*10^-6*sin(ω0*t); -0.3701336*10^-7*cos(ω0*t)];
    bw = [0 0; 0 0; 1/(Ix*ωn) 0; 0 1/(Iz*ωn) ]
    c = [1 0 0 0;0 1 0 0];
    d = zeros(2,1);
    dw = zeros(2,2);
    q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; r = [1.e-11;;]; s = zeros(4,1)

    # psysc = ps(a,convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d);
    # psyscw = ps(a,[convert(HarmonicArray,PeriodicFunctionMatrix(b,period)) bw],c,[d dw]);
    psysc = ps(a-0.01*I,PeriodicFunctionMatrix(b,period),c,d);
    psyscw = ps(a,[PeriodicFunctionMatrix(b,period) bw],c,[d dw]);


    K = 120;
    @time psysa = psc2d(psysc,period/K,reltol = 1.e-10);
    psys = convert(PeriodicStateSpace{PeriodicMatrix},psysa);
    @time psysaw = psc2d(psyscw,period/K,reltol = 1.e-10);
    psysw = convert(PeriodicStateSpace{PeriodicMatrix},psysaw);

    A = psys.A; B = psys.B; 
    q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; r = [1.e-11;;];
    Q = PeriodicMatrix(q, period; nperiod=K); R = PeriodicMatrix(r,period; nperiod=K);
    Qa = PeriodicArray(q, period; nperiod=K); Ra = PeriodicArray(r,period; nperiod=K);
    
    @time F, EVALS = pdlqr(psys, Q, R);
    ev = psceig(A+B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
    
    @time F, EVALS = pdlqr(psys, q, r);  
    ev = psceig(A+B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

    @time F, EVALS = pdlqr(psysa, Qa, Ra);
    ev = psceig(psysa.A+psysa.B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
    
    @time F, EVALS = pdlqr(psysa, q, r);  
    ev = psceig(psysa.A+psysa.B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

    qy = [2 0 ; 0 1]; r = [1.e-11;;];
    @time F, EVALS = pdlqry(psys, qy, r);  
    ev = psceig(A+B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
          
    @time F, EVALS = pdlqry(psysa, qy, r);  
    ev = psceig(psysa.A+psysa.B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
          
    Qy = PeriodicArray(qy, period; nperiod=K); Ra = PeriodicArray(r,period; nperiod=K); Sa = PeriodicArray(zeros(2,1),period; nperiod=K);
    @time F, EVALS = pdlqry(psysa, Qy, Ra);  
    ev = psceig(psysa.A+psysa.B*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

    @time F, EVALS = pdlqry(psysaw, qy, r, Sa);  
    ev = psceig(psysaw.A+psysaw.B[:,1:size(F,1)]*F)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6           
    
    qw = Matrix(I(4)); rv = 1.e-13*Matrix(I(2));       
    @time L, EVALS = pdkeg(psys, qw, rv; itmax = 2);  
    ev = psceig(psys.A-L*psys.C)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
       
    @time L, EVALS = pdkeg(psysa, qw, rv; itmax = 2);  
    ev = psceig(psysa.A-L*psysa.C)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 
      
    qw = 1.e-5*Matrix(I(2)); rv = 1.e-13*Matrix(I(2));       
    @time L, EVALS = pdkegw(psysw, qw, rv; itmax = 2);  
    ev = psceig(psysw.A-L*psysw.C)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-5 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-5 
       
    @time L, EVALS = pdkegw(psysaw, qw, rv; itmax = 2);  
    ev = psceig(psysaw.A-L*psysaw.C)
    @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-5 && 
          norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-5 
    
    PM = HarmonicArray
    PM = PeriodicFunctionMatrix
    for PM in (PeriodicFunctionMatrix, HarmonicArray)
        println("PM = $PM")
        psysc = ps(a,convert(PM,PeriodicFunctionMatrix(b,period)),c,d);
        #psyscw = ps(PM,dss(a,bw,c,dw),period);
        psyscw = ps(a,[convert(PM,PeriodicFunctionMatrix(b,period)) bw],c,[d dw]);
    
        @time F, EVALS = pclqr(psysc, q, r; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true); 
        clev = psceig(psysc.A+psysc.B*F,500)
        @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    
        # this test covers the experimental code provided in PeriodicSchurDecompositions package and occasionally fails
        @time F, EVALS = pclqr(psysc, q, r; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = false ); 
        clev = psceig(psysc.A+psysc.B*F,500)
        #println("EVALS = $EVALS, clev = $clev")
        @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    
        # low accuracy computation
        @time F, EVALS = pclqr(psysc, q, r; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = true); 
        clev = psceig(psysc.A+psysc.B*F,500)
        @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-1 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-1 

        @time F, EVALS = pclqry(psysc, qy, r; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true); 
        clev = psceig(psysc.A+psysc.B*F,500)
        @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7    

        qw = Matrix(I(4)); rv = 1.e-6*Matrix(I(2));       
        #qw = [0 0 0 0; 0 0 0 0; 0 0 0.05 0;0 0 0 1.e5]; rv = 1.e-13*Matrix(I(2));       
        @time L, EVALS = pckeg(psysc, qw, rv; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true);
        ev = psceig(psysc.A-L*psysc.C)
        @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
              norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

        qw = 1.e-4*Matrix(I(2)); rv = 1.e0*Matrix(I(2));       
        @time L, EVALS = pckegw(psyscw, qw, rv; K = 100, solver = "symplectic", intpol=true, reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true);  
        ev = psceig(psyscw.A-L*psyscw.C)
        @test norm(sort(real(EVALS))-sort(real(ev))) < 1.e-4 && 
              norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-4 

    end  
end
    

end

