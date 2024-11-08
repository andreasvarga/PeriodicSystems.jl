module Test_psconnect

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using LinearAlgebra: BlasInt
using ApproxFun
using MatrixPencils
#using BenchmarkTools

println("Test_psconnect")

@testset "test_psconnect" begin


# periodic time-varying matrices from IFAC2005 paper
@variables t
A = PeriodicSymbolicMatrix([cos(t) 1; 1 1-sin(t)],2*pi); 
B = PeriodicSymbolicMatrix([cos(t) + sin(t) 0; 1 1 - sin(t)],2*pi); 
C = PeriodicSymbolicMatrix([sin(2*t)+cos(2*t) 1; 1 0],2*pi); 
D = PeriodicSymbolicMatrix(Matrix{Int}(I(2)),pi)

psys = ps(A,B,C,D);
@test iszero(psteval(sin(t)^2*psys+psys*cos(t)^2-psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))


At = PeriodicFunctionMatrix(t->[cos(t) 1; 1 1-sin(t)],2*pi); 
Bt = PeriodicFunctionMatrix(t->[cos(t) + sin(t) 0; 1 1 - sin(t)],2*pi); 
Ct = PeriodicFunctionMatrix(t->[sin(2*t)+cos(2*t) 1; 1 0],2*pi); 
Dt = PeriodicFunctionMatrix(t->Matrix{Int}(I(2)),pi)
psys = ps(At,Bt,Ct,Dt);
@test iszero(psteval(psys+psys-2*psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))


psys = ps(convert(HarmonicArray,At), 
          convert(HarmonicArray,Bt), 
          convert(HarmonicArray,Ct), 
          convert(HarmonicArray,Dt));
@test iszero(psteval(psys+psys-2*psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))




K = 100
Δ = 2*pi/K
ts = (0:K-1)*Δ
period = promote_period(A,B,C,D)
Ats = PeriodicTimeSeriesMatrix(At.(ts),period);
Bts = PeriodicTimeSeriesMatrix(Bt.(ts),period);
Cts = PeriodicTimeSeriesMatrix(Ct(ts),period); 
Dts = PeriodicTimeSeriesMatrix(Dt.(ts),period);
psys = ps(Ats, Bts, Cts, Dts); 
@test iszero(psteval(psys+psys-2*psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))

Af = convert(FourierFunctionMatrix,At)
Bf = convert(FourierFunctionMatrix,Bt)
Cf = convert(FourierFunctionMatrix,Ct)
Df = convert(FourierFunctionMatrix,Dt)

psys = ps(Af,Bf,Cf,Df);
@test iszero(psteval(psys+psys-2*psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))

Asw = convert(PeriodicSwitchingMatrix,Ats)
Bsw = convert(PeriodicSwitchingMatrix,Bts)
Csw = convert(PeriodicSwitchingMatrix,Cts)
Dsw = convert(PeriodicSwitchingMatrix,Dts)
psys = ps(Asw,Bsw,Csw,Dsw);
@test iszero(psteval(psys+psys-2*psys,rand()))
psysi = psinv(psys)
@test iszero(psteval(psysi*psys,rand())-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))

                      
# constant dimensions
Ad = PeriodicMatrix([[1. 0; 0 0], [1 1;1 1], [0 1; 1 0]], 6, nperiod = 2);
Bd = PeriodicMatrix( [[ 1; 0 ], [ 1; 1]] ,2);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 0]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1);
psys = ps(Ad,Bd,Cd,Dd); 
@test iszero(ps2ls(psys+psys-2*psys)) 
psysi = psinv(psys)
@test iszero(ps2ls(psysi*psys)-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))


Adsw = convert(SwitchingPeriodicMatrix,Ad)
Bdsw = convert(SwitchingPeriodicMatrix,Bd)
Cdsw = convert(SwitchingPeriodicMatrix,Cd)
Ddsw = convert(SwitchingPeriodicMatrix,Dd)
psys = ps(Adsw,Bdsw,Cdsw,Ddsw);
@test iszero(ps2ls(psys+psys-2*psys)) 
psysi = psinv(psys)
@test iszero(ps2ls(psysi*psys)-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))


Ad = PeriodicMatrix([[1. 0], [1;1]],2);
Bd = PeriodicMatrix( [[ 1 ], [ 1; 1]] ,2);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 ]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1); 
psys = ps(Ad,Bd,Cd,Dd);
@test iszero(ps2ls(psys+psys-2*psys)) 
psysi = psinv(psys)
@test iszero(ps2ls(psysi*psys)-I)
@test iszero(psteval([psys psysi],1).A-[psteval(psys,1) psteval(psysi,1)].A)
@test iszero(psteval([psys; psysi],1).A-[psteval(psys,1); psteval(psysi,1)].A)
@test iszero(psteval(psappend(psys,psysi),1).A-append(psteval(psys,1), psteval(psysi,1)).A)


Ad = PeriodicArray(rand(Float32,2,2,10),10);
Bd = PeriodicArray(rand(2,1,2),2);
Cd = PeriodicArray(rand(1,2,3),3);
Dd = PeriodicArray(rand(1,1,1), 1);
psys = ps(Ad,Bd,Cd,Dd); 
@test iszero(ps2ls(psys+psys-2*psys)) 
psysi = psinv(psys)
@test iszero(ps2ls(psysi*psys)-I)
@test iszero(psteval([psys psysi],1)-[psteval(psys,1) psteval(psysi,1)])
@test iszero(psteval([psys; psysi],1)-[psteval(psys,1); psteval(psysi,1)])
@test iszero(psteval(psappend(psys,psysi),1)-append(psteval(psys,1), psteval(psysi,1)))

# Belgian chocolate problem  
# Bittanti-Colaneri, p. 35-38 
δ = 0.9
a = [0 1; -1 2δ]; b = [0;1;;]; c = [-2 2δ]; d = [1;;]
sys = dss(a,b,c,d)
sysd = c2d(sys,0.01)[1]

K1 = -4.22; K2 = 2.12; K = PeriodicSwitchingMatrix([[K1],[K2]],[0.,1.],2)
K0 = inv(I+K)*K

psyscl = psfeedback(sys, K0; negative=false)
eig = pspole(psyscl)
@test maximum(real.(eig)) < 0

sys = dss([ 1.49   0.49; 0.1   -1.57], [0.14;0.32;;], [ -0.37  0.66], [0.0;;]);
Kh = HarmonicArray([-82.03884329405999 + 0.0im;;; -30.207269256238863 - 159.28002605110427im], 1.0)
psyscl = psfeedback(sys, Kh; negative=false)
@test maximum(real(psceig(psyscl.A,120))) < 0.

sysd = c2d(sys,0.1)[1]
K =        
[[-148.26449538554698;;],
 [-234.96118359275238;;],
 [-275.1515476769106;;],
 [-253.48423457810625;;],
 [-178.23542145519684;;],
 [-78.14759730804454;;],
 [8.54909089916093;;],
 [48.73945498331912;;],
 [27.072141884514792;;],
 [-48.17667123839463;;]]
 Kp = PeriodicMatrix(K,1)
 psyscld = psfeedback(sysd, Kp; negative=false)
 @test maximum(abs.(pspole(psyscld))) ≈ 1.1831229767911728

Ad = 
[ -0.301204  -3.28208    0.356361
  0.677434  -0.951124  -0.557417
  0.962749  -0.903137   0.728291];
Bd =
[ 0.575855  -0.971467
0.727336   0.247178
1.25834    0.317415];
Cd = 
[ -0.258281  -1.01901   1.72766
-0.937695   0.855422  0.467289];
Dd = 
[ 0.988127  0.311045
0.650686  0.731904]; 
sysd = dss(Ad,Bd,Cd,Dd,Ts = 0.01)

Kd = 
[[0.1588328256338086 0.4805517499737685; -0.43236836920546506 0.2762320178288233],
[0.20277387007873166 0.48513648697814304; -0.357530989973105 0.2405182504169939],
[0.07344351569605928 0.5328175966999382; -0.392027148217488 0.23764479082499862]]
Kdopt = SwitchingPeriodicMatrix(Kd,[100, 200, 300],3.)

psyscld = psfeedback(sysd, Kdopt; negative=false)
@test maximum(abs.(pspole(psyscld))) < 1.


# error
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
psysc = ps(a-0.01*I,convert(HarmonicArray,PeriodicFunctionMatrix(b,period)),c,d);

K = HarmonicArray([0.6378318851574648 + 0.0im 0.026531410801766353 + 0.0im;;; 0.44872488693062773 + 0.7673482979266187im 0.25766667663046183 + 0.30595236856442165im;;; 0.7673482979266187 + 0.1377946642386344im 0.30595236856442165 + 0.39585017031175995im],psysc.period)
psyscl = psfeedback(psysc, K; negative=false); 
@test maximum(real(psceig(psyscl.A,20)))+0.01 < 1.e-5

end
end

