module Test_psutils

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using LinearAlgebra: BlasInt
using ApproxFun
using MatrixPencils
#using BenchmarkTools

println("Test_psutils")

@testset "test_psutils" begin

# periodic time-varying matrices from IFAC2005 paper
@variables t
A = PeriodicSymbolicMatrix([cos(t) 1; 1 1-sin(t)],2*pi); 
println("A(t) = $(A.F)")
B = HarmonicArray([0;1],[[1;0]],[[1;-1]],2*pi);
println("B(t) = $(convert(PeriodicSymbolicMatrix,B).F)")
C = PeriodicFunctionMatrix(t-> [sin(2*t)+cos(2*t) 1],pi);
println("C(t) = $(C.f(t))")
D = PeriodicFunctionMatrix(zeros(1,1),pi)
K = 100
Δ = 2*pi/K
ts = (0:K-1)*Δ

psys = ps(A,B,C,D);
psys = ps(A,B,C);
display(ps(D))
psys = ps(A,B,C,D,4*pi);
psys = ps(A,B,C,4*pi);
psys = ps(HarmonicArray,A,B,C,D,4*pi);
psys = ps(PeriodicSymbolicMatrix, A,B,C,4*pi);

psys = ps(rss(4,3,2),10)
@test islti(psys)
@test lastindex(psys,1) == 3 && lastindex(psys,2) == 2
@test iszero(psaverage(psys)[2:3,1]-psaverage(psys[2:3,1]))


psys=ps(rand(2,2),rand(2),rand(3,2),rand(3),10)
psys=ps(rand(2,2),rand(2),rand(3,2),rand(3),10;Ts = 1)
psys=ps(rand(2,2),rand(2),rand(3,2),10;Ts = 1)

psys = PeriodicStateSpace(convert(PeriodicFunctionMatrix,A), 
                          convert(PeriodicFunctionMatrix,B), 
                          convert(PeriodicFunctionMatrix,C), 
                          convert(PeriodicFunctionMatrix,D));
print(psys);
convert(PeriodicStateSpace{PeriodicSymbolicMatrix},psys);
convert(PeriodicStateSpace{HarmonicArray},psys);
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);
convert(PeriodicStateSpace{FourierFunctionMatrix},psys);

psys = PeriodicStateSpace(convert(PeriodicFunctionMatrix{:c,BigFloat},A), 
                          convert(PeriodicFunctionMatrix{:c,BigFloat},B), 
                          convert(PeriodicFunctionMatrix{:c,BigFloat},C), 
                          convert(PeriodicFunctionMatrix{:c,BigFloat},D));

psys = ps(convert(PeriodicFunctionMatrix,A), 
          convert(PeriodicFunctionMatrix,B), 
          convert(PeriodicFunctionMatrix,C), 
          convert(PeriodicFunctionMatrix,D));

psys = ps(convert(PeriodicFunctionMatrix,A), 
          convert(PeriodicFunctionMatrix,B), 
          convert(PeriodicFunctionMatrix,C));


psys = PeriodicStateSpace(convert(PeriodicSymbolicMatrix,A), 
                          convert(PeriodicSymbolicMatrix,B), 
                          convert(PeriodicSymbolicMatrix,C), 
                          convert(PeriodicSymbolicMatrix,D));
print(psys);
convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys);
convert(PeriodicStateSpace{HarmonicArray},psys);
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);
convert(PeriodicStateSpace{FourierFunctionMatrix},psys);



psys = ps(convert(PeriodicSymbolicMatrix,A), 
          convert(PeriodicSymbolicMatrix,B), 
          convert(PeriodicSymbolicMatrix,C), 
          convert(PeriodicSymbolicMatrix,D));

psys = ps(convert(PeriodicSymbolicMatrix,A), 
          convert(PeriodicSymbolicMatrix,B), 
          convert(PeriodicSymbolicMatrix,C));


psys = PeriodicStateSpace(convert(HarmonicArray,A), 
                          convert(HarmonicArray,B), 
                          convert(HarmonicArray,C), 
                          convert(HarmonicArray,D));
print(psys);
print(ps(psys.D))
convert(PeriodicStateSpace{PeriodicSymbolicMatrix},psys);
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);
convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys);
convert(PeriodicStateSpace{FourierFunctionMatrix},psys);

psys = ps(convert(HarmonicArray,A), 
          convert(HarmonicArray,B), 
          convert(HarmonicArray,C), 
          convert(HarmonicArray,D));

psys = ps(convert(HarmonicArray,A), 
          convert(HarmonicArray,B), 
          convert(HarmonicArray,C));



At = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,A).f.(ts),A.period);
Bt = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,B).f.(ts),B.period);
Ct = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,C).f.(ts),C.period); 
Dt = PeriodicTimeSeriesMatrix(convert(PeriodicFunctionMatrix,D).f.(ts),D.period);
psys = PeriodicStateSpace(At, Bt, Ct, Dt); 
print(psys);
print(ps(psys.D))
convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys);
convert(PeriodicStateSpace{PeriodicSymbolicMatrix},psys);
convert(PeriodicStateSpace{HarmonicArray},psys);
convert(PeriodicStateSpace{FourierFunctionMatrix},psys);

psys = ps(At, Bt, Ct, Dt);
psys = ps(At, Bt, Ct);


psys = PeriodicStateSpace(convert(FourierFunctionMatrix,A), 
                          convert(FourierFunctionMatrix,B), 
                          convert(FourierFunctionMatrix,C), 
                          convert(FourierFunctionMatrix,D));
print(psys);
@test_throws "not available" print(ps(psys.D))
convert(PeriodicStateSpace{PeriodicFunctionMatrix},psys);
convert(PeriodicStateSpace{HarmonicArray},psys); 
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);
convert(PeriodicStateSpace{PeriodicSymbolicMatrix},psys);

psys = ps(convert(FourierFunctionMatrix,A), 
          convert(FourierFunctionMatrix,B), 
          convert(FourierFunctionMatrix,C), 
          convert(FourierFunctionMatrix,D));
psys = ps(convert(FourierFunctionMatrix,A), 
          convert(FourierFunctionMatrix,B), 
          convert(FourierFunctionMatrix,C));

psys = ps(FourierFunctionMatrix,A,B,C,D);                          

# #new tests
# AF = ffm2hr(psys.A)
# convert(PeriodicSymbolicMatrix,AF) 

# constant dimensions
Ad = PeriodicMatrix([[1. 0; 0 0], [1 1;1 1], [0 1; 1 0]], 6, nperiod = 2);
Bd = PeriodicMatrix( [[ 1; 0 ], [ 1; 1]] ,2);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 0]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1);
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd); 
convert(PeriodicStateSpace{PeriodicArray},psys);
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);

psys = ps(Ad,Bd,Cd,Dd);
psys = ps(Ad,Bd,Cd); 


Ad = PeriodicMatrix([[1. 0], [1;1]],2);
Bd = PeriodicMatrix( [[ 1 ], [ 1; 1]] ,2);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 ]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1); 
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd);
print(psys);
print(ps(psys.D))
convert(PeriodicStateSpace{PeriodicArray},psys);
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys);

psys = ps(Ad,Bd,Cd,Dd);
psys = ps(Ad,Bd,Cd); 


Ad = PeriodicArray(rand(Float32,2,2,10),10);
Bd = PeriodicArray(rand(2,1,2),2);
Cd = PeriodicArray(rand(1,2,3),3);
Dd = PeriodicArray(rand(1,1,1), 1);
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd); 
print(psys);
print(ps(psys.D))
convert(PeriodicStateSpace{PeriodicMatrix},psys)
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys)

psys = ps(Ad,Bd,Cd,Dd);
psys = ps(Ad,Bd,Cd); 
psys = ps(convert(PeriodicMatrix,Ad), Bd, convert(PeriodicMatrix,Cd), Dd); 
psys = ps(convert(PeriodicMatrix,Ad), Bd, convert(PeriodicMatrix,Cd)); 

print(ps(SwitchingPeriodicArray(rand(1,2,3),[2,4,6],10)))

print(ps(SwitchingPeriodicMatrix(PeriodicMatrices.pmzeros([1,1],[2,2]),[5,10],10)))
end
end

