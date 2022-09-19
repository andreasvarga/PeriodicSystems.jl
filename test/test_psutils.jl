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
ps(A);
ps(B);
ps(C);
ps(D);
psys = ps(A,B,C,D,4*pi);
psys = ps(A,B,C,4*pi);
psys = ps(HarmonicArray,A,B,C,D,4*pi);
psys = ps(PeriodicSymbolicMatrix, A,B,C,4*pi);


psys = ps(rss(4,3,2),10)
@test islti(psys)
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
convert(PeriodicStateSpace{PeriodicMatrix},psys)
convert(PeriodicStateSpace{PeriodicTimeSeriesMatrix},psys)

psys = ps(Ad,Bd,Cd,Dd);
psys = ps(Ad,Bd,Cd); 
psys = ps(convert(PeriodicMatrix,Ad), Bd, convert(PeriodicMatrix,Cd), Dd); 
psys = ps(convert(PeriodicMatrix,Ad), Bd, convert(PeriodicMatrix,Cd)); 


# symbolic periodic 
@variables t
A = [cos(t) 1; 1 1-sin(t)];
B = [cos(t)+sin(t); 1-sin(t)];
C = [sin(t)+cos(2*t) 1];
Ap = PeriodicSymbolicMatrix(A,2*pi);
Bp = PeriodicSymbolicMatrix(B,2*pi);
Cp = PeriodicSymbolicMatrix(C,2*pi);

# functional expressions
tA(t::Real) = [cos(t) 1; 1 1-sin(t)];
tB(t::Real) = [cos(t)+sin(t); 1-sin(t)];
tC(t::Real) = [sin(t)+cos(2*t) 1];
# store snapshots as 3d arrays
N = 200; 
tg = collect((0:N-1)*2*pi/N);
time = (0:N-1)*2*pi/N;
# At = reshape(hcat(A.(t)...),2,2,N);  
# Bt = reshape(hcat(B.(t)...),2,1,N);  
# Ct = reshape(hcat(C.(t)...),1,2,N);

# time series expressions
At = PeriodicTimeSeriesMatrix(tA.(time),2*pi);
Bt = PeriodicTimeSeriesMatrix(tB.(time),2*pi);
Ct = PeriodicTimeSeriesMatrix(tC.(time),2*pi);

# harmonic expressions
@time Ahr = ts2hr(At);
@test Ahr.values[:,:,1] ≈ [0. 1; 1 1] && Ahr.values[:,:,2] ≈ [1. 0; 0 -im] 
@time Bhr = ts2hr(Bt);
@test Bhr.values[:,:,1] ≈ [0.; 1] && Bhr.values[:,:,2] ≈ [1.0+im; -im]
@time Chr = ts2hr(Ct);
@test Chr.values[:,:,1] ≈ [0. 1] && Chr.values[:,:,2] ≈ [im 0] && Chr.values[:,:,3] ≈ [1 0]

#@time Affm = ts2ffm(At);

nperiod = 24
time1 = (0:N-1)*2*pi*nperiod/N;
At1 = PeriodicTimeSeriesMatrix(tA.(time1),2*pi*nperiod);
Ahr1 = ts2hr(At1);
@test convert(PeriodicFunctionMatrix,Ahr1).f(1) ≈ convert(PeriodicFunctionMatrix,Ahr).f(1) 

@test iszero(hr2psm(Ahr,1:1) + hr2psm(Ahr,0:0) - hr2psm(Ahr))
@test iszero(hr2psm(Ahr1,1:1) + hr2psm(Ahr1,0:0) - hr2psm(Ahr1))

# harmonic vs. symbolic
@test norm(substitute.(convert(PeriodicSymbolicMatrix,Ahr).F - A, (Dict(t => rand()),))) < 1e-15
@test norm(substitute.(convert(PeriodicSymbolicMatrix,Bhr).F - B, (Dict(t => rand()),))) < 1e-15
@test norm(substitute.(convert(PeriodicSymbolicMatrix,Chr).F - C, (Dict(t => rand()),))) < 1e-15

# harmonic vs. time series
@test all(norm.(tvmeval(Ahr,tg).-At.values) .< 1.e-7)
@test all(norm.(tvmeval(Bhr,tg).-Bt.values) .< 1.e-7)
@test all(norm.(tvmeval(Chr,tg).-Ct.values) .< 1.e-7)

# check time values on the grid
for method in ("constant", "linear", "quadratic", "cubic")
      @test all(norm.(tvmeval(At,tg; method).-At.values) .< 1.e-7)
      @test all(norm.(tvmeval(Bt,tg; method).-Bt.values) .< 1.e-7)
      @test all(norm.(tvmeval(Ct,tg; method).-Ct.values) .< 1.e-7)
end      
# check interpolated values: time series vs. harmonic
tt = rand(10)*2pi;
for method in ("linear", "quadratic", "cubic")
    @test all(norm.(tvmeval(At,tt; method).-tvmeval(Ahr,tt; exact = true)) .< 1.e-3)
    @test all(norm.(tvmeval(Bt,tt; method).-tvmeval(Bhr,tt; exact = false)) .< 1.e-3)
    @test all(norm.(tvmeval(Ct,tt; method).-tvmeval(Chr,tt; exact = true)) .< 1.e-3)
end

# check conversion to function form
Amat = convert(PeriodicFunctionMatrix,Ahr); 
@test all(norm.(tvmeval(At,tt; method = "linear").-tvmeval(Ahr,tt)) .< 1.e-3)
@test iszero(convert(PeriodicSymbolicMatrix,Amat).F-Ap.F)
@test isperiodic(Amat)

Amat =  convert(PeriodicFunctionMatrix,At);
@test all(norm.(tvmeval(At,tt; method = "linear").-Amat.f.(tt)) .< 1.e-3)
#@test iszero(convert(PeriodicSymbolicMatrix,Amat).F-Ap.F)
@test isperiodic(Amat)

Amat = PeriodicFunctionMatrix(tA,2pi);
@test all(norm.(tvmeval(At,tt; method = "linear").-tvmeval(Amat,tt)) .< 1.e-3)
@test iszero(convert(PeriodicSymbolicMatrix,Amat).F-Ap.F)
@test isperiodic(Amat)

Amat =  convert(PeriodicFunctionMatrix,Ap);
@test all(norm.(tvmeval(At,tt; method = "linear").-tvmeval(Ap,tt)) .< 1.e-3)
@test iszero(convert(PeriodicSymbolicMatrix,Amat).F-Ap.F)
@test isperiodic(Amat) && isperiodic(Ap) && size(Amat) == size(Ap)


for method in ("constant", "linear", "quadratic", "cubic")
      Amat = ts2pfm(At; method);
      @test all(norm.(At.values.-Amat.f.(tg)) .< 1.e-10)
      Bmat = ts2pfm(Bt; method);
      @test all(norm.(Bt.values.-Bmat.f.(tg)) .< 1.e-10)
      Cmat = ts2pfm(Ct; method);
      @test all(norm.(Ct.values.-Cmat.f.(tg)) .< 1.e-10)
end

# example of Colaneri
at(t) = [0 1; -10*cos(t) -24-10*sin(t)];
Afun=PeriodicFunctionMatrix(at,2pi); 
ev = pseig(Afun; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
# @time Φ = tvstm(Afun.f, Afun.period; reltol = 1.e-10)
cvals = log.(complex(ev))/2/pi 
println("cvals = $cvals  -- No one digit accuracy!!!")
@test maximum(abs.(ev)) ≈ 1

# using ApproxFun
s = Fourier(0..2π)
At = FourierFunctionMatrix(Fun(t -> [0 1; -10*cos(t) -24-10*sin(t)],s), 2pi)
Atfun = convert(PeriodicFunctionMatrix,At)
Ahrfun = convert(PeriodicFunctionMatrix,pfm2hr(Afun))

@time cvals = psceig(Afun, 500; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
@time cvals1 = psceig(Atfun, 500; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
@time cvals2 = psceig(Ahrfun, 500; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
@test cvals ≈ cvals1 ≈ cvals2
# Tt = Fun(t -> [12+5*sin(t) 1/2; 1 0],s)
# Tinvt=inv(Tt)
# Atilde=Tt*At.M*Tinvt+Tt'*Tinvt
# Aref = Fun(t -> [0 0; 2 -24-10*sin(t)],s)
# @test norm(Aref-Atilde) < 1.e-10

# example Floquet analysis from ApproxFun.jl
a=0.15
at1(t) = -[0 -1 0 0; (2+a*cos(2t)) 0 -1 0; 0 0 0 -1; -1 0 (2+a*cos(2t)) 0]
Afun1=PeriodicFunctionMatrix(at1,pi);
ev1 = pseig(Afun1; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
cvals1 = log.(complex(ev1))/pi

a=0.15
at2(t) = -[0 -1 0 0; (2+a*cos(2t)) 0 -1 0; 0 0 0 -1; -1 0 (2+a*cos(2t)) 0]
Afun2=PeriodicFunctionMatrix(at2,2*pi;nperiod=2);
ev2 = pseig(Afun2; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10)
cvals2 = log.(complex(ev2))/(2pi)
@test ev1.^2 ≈ ev2 && real(cvals1) ≈ real(cvals2)


# full accuracy characteristic exponents
# solver = "symplectic"
# for solver in ("non-stiff", "stiff", "linear", "symplectic", "noidea")
#     @time M = monodromy(Afun, 500; solver, reltol = 1.e-10, abstol = 1.e-10);
#     cvals = log.(complex(pseig(M)))/2/pi
#     #println("solver = $solver cvals = $cvals")
#     @test isapprox(cvals, [0; -24], atol = 1.e-7)
# end

solver = "non-stiff"
#for solver in ("non-stiff", "stiff", "linear", "symplectic", "noidea")
for solver in ("non-stiff", "stiff", "linear", "noidea")
      println("solver = $solver")
    @time cvals = psceig(Afun, 500; solver, reltol = 1.e-10, abstol = 1.e-10)
    @test isapprox(cvals, [0; -24], atol = 1.e-7)
end

# Vinograd example: unstable periodic system with all A(t) stable
#at(t) = [3.5 6;-6 -5.5]+[-4.5 6; 6 4.5]*cos(12*t)+[6 4.5;4.5 -6]*sin(12*t); T = pi/6; # does not work
function at(t::Real)
    [-1-9*(cos(6*t))^2+12*sin(6*t)*cos(6*t) 12*(cos(6*t))^2+9*sin(6*t)*cos(6*t);
    -12*(sin(6*t))^2+9*sin(6*t)*cos(6*t) -1-9*(sin(6*t))^2-12*sin(6*t)*cos(6*t)]
end
@test eigvals(at(rand())) ≈ [-10,-1]
T = pi/3;
Afun=PeriodicFunctionMatrix(at,T);
#for solver in ("non-stiff", "stiff", "linear", "symplectic", "noidea")
for solver in ("non-stiff", "stiff", "linear", "noidea")
      @time cvals = psceig(Afun, 500; solver, reltol = 1.e-10)
      @test cvals ≈ [2; -13]
end  


end
end

