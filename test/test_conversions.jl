module Test_conversions

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra

println("Test_conversions")

@testset "psmrc2d" begin

# example Longhi IEEETAC, 1994
sys = dss([0 -pi/6; pi/6 0],eye(2),eye(2),0)
@time psys = psmrc2d(sys,1; ki = [2,3], ko = [1,2])
@test psys.period == 6

@time psys = psmrc2d(sys,1);
@test psys.period == 1

@time psys = psmrc2d(sys,1, ki = [2,3]);
@test psys.period == 6

@time psys = psmrc2d(sys,1, ko = [1,2]);
@test psys.period == 2

sys = rss(2,2,2,disc=true)
@time psys = psmrc2d(sys,1; ki = [2,3], ko = [1,2])
@test psys.period == 6

end 

@testset "psc2d" begin

# periodic time-varying matrices from IFAC2005 paper
Ta = 4*pi; Tb = 2*pi; Tc = 8*pi; Td = pi; 
@variables t
A = PeriodicSymbolicMatrix([cos(t) 1; 1 1-sin(t)],Ta); 
println("A(t) = $(A.F)")
B = HarmonicArray([0;1],[[1;0]],[[1;-1]],Tb);
println("B(t) = $(convert(PeriodicSymbolicMatrix,B).F)")
C = PeriodicFunctionMatrix(t-> [sin(2*t)+cos(2*t) 1],Tc);
println("C(t) = $(C.f(t))")
D = PeriodicFunctionMatrix(zeros(1,1),Td)
K = 200
Δa = Ta/K; Δb = Tb/K; Δd = Td/K; Δc = Tc/K;

tsa = (0:K-1)*Δa
ts = (0:K-1)*Δb
tsd = (0:K-1)*Δd
tsc = (0:K-1)*Δc
Δ = Δb

psysc = PeriodicStateSpace(convert(PeriodicFunctionMatrix,A), 
                          convert(PeriodicFunctionMatrix,B), 
                          convert(PeriodicFunctionMatrix,C), 
                          convert(PeriodicFunctionMatrix,D));

@time psys = psc2d(psysc,Δ);
@time sys = psaverage(psysc); 


psysc1 = PeriodicStateSpace(convert(PeriodicSymbolicMatrix,A), 
                          convert(PeriodicSymbolicMatrix,B), 
                          convert(PeriodicSymbolicMatrix,C), 
                          convert(PeriodicSymbolicMatrix,D));
@time psys1 = psc2d(psysc1,Δ);
@test norm(psys.A-psys1.A,Inf) < 1.e-7 && norm(psys.B-psys1.B,Inf) < 1.e-7 && 
      norm(psys.C-psys1.C,Inf) < 1.e-7 && norm(psys.D-psys1.D,Inf) < 1.e-7
@time sys1 = psaverage(psysc1); 
@test iszero(sys-sys1,atol=1.e-7)


psysc2 = PeriodicStateSpace(convert(HarmonicArray,A), 
                          convert(HarmonicArray,B), 
                          convert(HarmonicArray,C), 
                          convert(HarmonicArray,D));
@time psys2 = psc2d(psysc2,Δ);
@test norm(psys.A-psys2.A,Inf) < 1.e-7 && norm(psys.B-psys2.B,Inf) < 1.e-7 && 
      norm(psys.C-psys2.C,Inf) < 1.e-7 && norm(psys.D-psys2.D,Inf) < 1.e-7
@time sys2 = psaverage(psysc2); 
@test iszero(sys-sys2,atol=1.e-7)

psysc3 = PeriodicStateSpace(convert(FourierFunctionMatrix,A), 
                          convert(FourierFunctionMatrix,B), 
                          convert(FourierFunctionMatrix,C), 
                          convert(FourierFunctionMatrix,D));
@time psys3 = psc2d(psysc3,Δ);
@test norm(psys.A-psys3.A,Inf) < 1.e-7 && norm(psys.B-psys3.B,Inf) < 1.e-7 && 
      norm(psys.C-psys3.C,Inf) < 1.e-7 && norm(psys.D-psys3.D,Inf) < 1.e-7
@time sys3 = psaverage(psysc3); 
@test iszero(sys-sys3,atol=1.e-7)


At = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,A).f.(tsa),A.period);
Bt = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,B).f.(ts),B.period);
Ct = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,C).f.(tsc),C.period); 
Dt = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,D).f.(tsd),D.period);


psysc4 = PeriodicStateSpace(At, Bt, Ct, Dt); 
@time psys4 = psc2d(psysc4,Δ);
@test norm(psys.A - psys4.A,Inf) < 1.e-7 && norm(psys.B - psys4.B) < 1.e-7 && 
      norm(psys.C-psys4.C) < 1.e-7 && norm(psys.D-psys4.D) < 1.e-7
@time sys4 = psaverage(psysc4); 
@test iszero(sys-sys4,atol=1.e-7)


A1=convert(PeriodicTimeSeriesMatrix,A)
B1=convert(PeriodicTimeSeriesMatrix,B)
C1=convert(PeriodicTimeSeriesMatrix,C)
D1=convert(PeriodicTimeSeriesMatrix,D)
psysc5 = PeriodicStateSpace(A1, B1, C1, D1); 
@time psys5 = psc2d(psysc5,Δ);
@test norm(psys.A - psys5.A,Inf) < 1.e-7 && norm(psys.B - psys5.B,Inf) < 1.e-7 && 
      norm(psys.C-psys5.C,Inf) < 1.e-7 && norm(psys.D-psys5.D,Inf) < 1.e-7


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

K = 120;
@time psys = psc2d(psysc,period/K,reltol = 1.e-10);
@test all(abs.(pseig(psys.A)) .≈ 1)
@test psys.A.M[1] ≈ exp(a*period/K) && psys.C.M[1] == c && psys.D.M[1] == d

# lti system 
sys = rss(3,2,3);
psysc = ps(sys,10);
@time sysd = c2d(sys,1)[1];
@time psys = psc2d(psysc,1);
@test psys.A.M[1] ≈ sysd.A && psys.B.M[1] == sysd.B && psys.C.M[1] == sysd.C && psys.D.M[1] == sysd.D

# only dynamic gains
psysc = ps(C);
Ts = pi/10;
@time psys = psc2d(psysc,Ts);
@test all([psys.D.M[i] == C.f((i-1)*Ts) for i in 1:10])

# A and B constant matrices
a = rand(2,2); b = rand(2,1); c = rand(1,2); d = rand(1,1);
sys = dss(a,b,c,d);
psysc = ps(a,b,C,d);
Ts = pi/10;
@time sysd = c2d(sys,Ts)[1];
@time psys = psc2d(psysc,Ts);
@test psys.A.M[1] ≈ sysd.A && psys.B.M[1] == sysd.B && 
      all([psys.C.M[i] == C.f((i-1)*Ts) for i in 1:10]) && psys.D.M[1] == sysd.D

end # psc2d      
end # module