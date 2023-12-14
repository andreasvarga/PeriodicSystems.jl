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


println("Test_stabilization")

@testset "test_output_feedback_stabilization" begin


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
@time psys = psc2d(psysc,period/K,reltol = 1.e-10);
psysa = convert(PeriodicStateSpace{PeriodicArray},psys);

#Q = eye(4); R = eye(1)*1.e-7; 
Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time F, info = plqofc(psys, Q, R);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5 
@time F1, info1 = plqofc(psys, Q, R,zeros(4,1));
@time F2, info2 = plqofc(psys, Q, R,PeriodicMatrix(zeros(4,1),psys.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt

Qp = PeriodicMatrix([Q],period); Rp = PeriodicMatrix([R],period); 
Sp = PeriodicMatrix([zeros(4,1)],period);
@time Fp, infop = plqofc(psys, Qp, Rp, Sp);
@test info.fopt ≈ infop.fopt 

for k in (1, 2, 3, 10, 60, 120)
    @time Fsw, infosw = plqofc_sw(psys, Q, R, ns = collect(k:k:120));
    psyscl = psfeedback(psys, Fsw; negative=false); 
    @test abs(maximum(abs.(pseig(psyscl.A)))-infosw.sdeg) < 1.e-5
    println(" k = $k,  sdeg = $(infosw.sdeg), Jopt = $(infosw.fopt)")
    k == 1 && (@test info.fopt ≈ infosw.fopt) 
end



Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time Fa, info = plqofc(psysa, Q, R);
psyscl = psfeedback(psysa, Fa; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5
@time F1, info1 = plqofc(psysa, Q, R,zeros(4,1));
@time F2, info2 = plqofc(psysa, Q, R,PeriodicArray(zeros(4,1),psys.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt
@test norm(convert(PeriodicArray,F)-Fa) < 1.e-7

Qa = PeriodicArray(Q,period); Ra = PeriodicArray(R,period); 
Sa = PeriodicArray(zeros(4,1),period);
@time Fa, infoa = plqofc(psysa, Qa, Ra, Sa);
@test info.fopt ≈ infoa.fopt 

for k in (1, 2, 3, 10, 60, 120)
    @time Fasw, infosw = plqofc_sw(psysa, Q, R, ns = collect(k:k:120));
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
@time psys1 = psc2d(psysc1,period/K,reltol = 1.e-10);
psysa1 = convert(PeriodicStateSpace{PeriodicArray},psys1);

#Q = eye(4); R = eye(1)*1.e-7; 
Q = diagm([2,1,0,0]);  R = eye(1)*1.e-11;
@time F, info = plqofc(psys1, Q, R);
psyscl = psfeedback(psys1, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5 
@time F1, info1 = plqofc(psys1, Q, R,zeros(4,1));
@time F2, info2 = plqofc(psys1, Q, R,PeriodicMatrix(zeros(4,1),psys1.period));
@test info.fopt ≈ info1.fopt ≈ info2.fopt


@time Fa, infoa = plqofc(psysa1, Q, R);
psyscl = psfeedback(psysa1, Fa; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5
@time F1, infoa1 = plqofc(psysa1, Q, R,zeros(4,1));
@time F2, infoa2 = plqofc(psysa1, Q, R,PeriodicArray(zeros(4,1),psys1.period));
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
psys = psc2d(psysc,period/K,reltol = 1.e-10);
psysa = convert(PeriodicStateSpace{PeriodicArray},psys);
@time F, info = plqofc(psys, Q, R, gtol=0.01,lub = (-5000,5000),maxiter=100);
@time Fa, infoa = plqofc(psysa, Q, R, gtol=0.01,lub = (-5000,5000),maxiter=100);
@test info.fopt ≈ infoa.fopt 


for K in (10, 20, 40, 120)
    psys = psc2d(psysc,period/K,reltol = 1.e-10);
    psysa = convert(PeriodicStateSpace{PeriodicArray},psys);

    @time F, info = plqofc(psys, Q, R, gtol=0.0001);
    @time Fa, infoa = plqofc(psysa, Q, R, gtol=0.0001);
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
@time F, info = plqofc(psys, Q, R, gtol=0.01);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5

Q = PeriodicMatrix([eye(2),eye(1)],2); R = PeriodicMatrix([eye(1),2*eye(1)],2); 
S = PeriodicMatrix([0.001*ones(1,1),0.001*ones(2,1)],2);
@time F, info = plqofc(psys, Q, R, S, gtol=0.01);
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5

@time F, info = plqofc(psys, Q, R, S, gtol=0.01, optimizer = NelderMead());
psyscl = psfeedback(psys, F; negative=false); 
@test abs(maximum(abs.(pseig(psyscl.A)))-info.sdeg) < 1.e-5



end
end

