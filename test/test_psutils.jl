module Test_psutils

using PeriodicSystems
using DescriptorSystems
using Interpolations
#using DifferentialEquations
using OrdinaryDiffEq
using Symbolics
using FFTW
#using SLICOTMath
#using SLICOTtools
using LinearAlgebra
using LinearAlgebra: BlasInt
using Test
#using StaticArrays
using IRKGaussLegendre
using Primes: factor
using ApproxFun
#using ApproxFun
#using BenchmarkTools

println("Test_psutils")

@testset "test_psutils" begin



# MB03BD example
A1 = Matrix{Float64}(I,3,3); A2 = [   1.0   2.0   0.0; 4.0  -1.0   3.0; 0.0   3.0   1.0]; A3 = Matrix{Float64}(I,3,3); 
E1 =  [2.0   0.0   1.0; 0.0  -2.0  -1.0; 0.0   0.0   3.0]; E2 = Matrix{Float64}(I,3,3); 
E3 = [ 1.0   0.0   1.0; 0.0   4.0  -1.0; 0.0   0.0  -2.0];
ev = eigvals(inv(E1)*A2*inv(E3))

# using the SLICOT wrapper
A = reshape([E1 A2 E3],3,3,3);
KMAX = 3
NMAX = 3
LDA1 = NMAX
LDA2 = NMAX
LDQ1 = NMAX
LDQ2 = NMAX
LDWORK = KMAX + max( 2*NMAX, 8*KMAX )
LIWORK = 2*KMAX + NMAX
QIND = Array{BlasInt,1}(undef, KMAX)
S = [-1,1,-1]; 
Q = Array{Float64,3}(undef, LDQ1, LDQ2, KMAX)
ALPHAR = Array{Float64,1}(undef, NMAX)
ALPHAI = Array{Float64,1}(undef, NMAX)
BETA = Array{Float64,1}(undef, NMAX)
SCAL = Array{BlasInt,1}(undef, NMAX)
IWORK = Array{BlasInt,1}(undef, LIWORK)
DWORK = Array{Float64,1}(undef, LDWORK)

mb03bd!('T','C','I',QIND,3,3,2,1,3,S,A,Q,ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)

poles = (ALPHAR+im*ALPHAI) ./ BETA .* (2. .^SCAL)

@test sort(real(poles)) ≈ sort(real(ev)) && 
      sort(imag(poles)) ≈ sort(imag(ev))

A = reshape([A2 inv(E3) inv(E1)],3,3,3);
mb03wd!('S','I',3,3,1,3,1,3,A,Q,ALPHAR, ALPHAI, LDWORK)

poles = (ALPHAR+im*ALPHAI) 

@test sort(real(poles)) ≈ sort(real(ev)) && 
      sort(imag(poles)) ≈ sort(imag(ev))




# MB03VD example
A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
A2 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.];
n = 4; K = 2; 
A = reshape([A1 A2], n, n, K);
@time H, Z, ihess = phess(A; rev = false);
@test check_psim(A,Z,H; rev = false) && istriu(H[:,:,ihess],-1) && ihess == 1

@time H1, Z1, ihess1 = phess(A; rev = false, withZ = false);
@test H[1] == H1[1] && H[2] == H1[2] 

@time H, Z, ihess = phess(A, hind = 2, rev = false);
@test check_psim(A,Z,H; rev = false) && istriu(H[:,:,ihess],-1) && ihess == 2

@time H1, Z1, ihess1 = phess(A, hind = 2, rev = false, withZ = false);
@test H[1] == H1[1] && H[2] == H1[2]  

@time H, Z, ihess = phess(A; rev = true); 
@test check_psim(A,Z,H; rev = true) && istriu(H[:,:,ihess],-1) && ihess == 1

@time H1, Z1, ihess1 = phess(A; rev = true, withZ = false); 
@test H[1] == H1[1] && H[2] == H1[2]  

@time H, Z, ihess = phess(A; hind = 2, rev = true);
@test check_psim(A,Z,H; rev = true) && istriu(H[:,:,ihess],-1) && ihess == 2

@time H1, Z1, ihess1 = phess(A; hind = 2, rev = true, withZ = false);
@test H[1] == H1[1] && H[2] == H1[2] 

ev = eigvals(A[:,:,1]*A[:,:,2])
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
@test check_psim(A,Z,S; rev = false) && istriu(S[:,:,ischur],-1) && ischur == 1
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

@time S1, Z1, eigs1, ischur1, α1, γ1 = pschur(A; rev = false, withZ = false);
@test S[1] == S1[1] && S[2] == S1[2] && eigs == eigs1


@time S, Z, eigs, ischur, α, γ = pschur(A, sind = 2, rev = false); 
@test check_psim(A,Z,S; rev = false) && istriu(S[:,:,ischur],-1) && ischur == 2
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

ev = eigvals(A[:,:,2]*A[:,:,1])
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = true); 
@test check_psim(A,Z,S; rev = true) && istriu(S[:,:,ischur],-1) && ischur == 1
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

@time S, Z, eigs, ischur, α, γ = pschur(A; sind = 2, rev = true); 
@test check_psim(A,Z,S; rev = true) && istriu(S[:,:,ischur],-1) && ischur == 2
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
n = 4; K = 18;
A = reshape(hcat([A1 for i in 1:K]...), n, n, K);
@time H, Z, ihess = phess(A; rev = false);
@test check_psim(A,Z,H; rev = false) && istriu(H[:,:,ihess],-1) && ihess == 1

@time S, Z, eigs, ischur, = pschur(A; rev = false);
@test check_psim(A,Z,S; rev = false)

# @time S1, Z1, eigs1, ischur1 = pschurw(A; rev = false);
# @test check_psim(A,Z1,S1; rev = false)

A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
ev1 = eigvals(A1)
n = 4; K = 18;

k = 1; N = div(K,k)
Ap = PeriodicArray(reshape(hcat([A1 for i in 1:N]...), n, n, N), K; nperiod = k)
@time ev = pseig(Ap)
@test sort(real(ev)) ≈ sort(real(ev1.^K)) && sort(imag(ev)) ≈ sort(imag(ev1.^K))

k = 9; N = div(K,k)
Ap = PeriodicArray(reshape(hcat([A1 for i in 1:N]...), n, n, N),K; nperiod = k)
@time ev = pseig(Ap)
@test sort(real(ev)) ≈ sort(real(ev1.^K)) && sort(imag(ev)) ≈ sort(imag(ev1.^K))

k = 18; N = div(K,k)
Ap = PeriodicArray(reshape(hcat([A1 for i in 1:N]...), n, n, N),K; nperiod = k)
@time ev = pseig(Ap)
@test sort(real(ev)) ≈ sort(real(ev1.^K)) && sort(imag(ev)) ≈ sort(imag(ev1.^K))




# modified MB03VD example
A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
A2 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.] .+ 1;
A3 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.] .- 1;

n = 4; K = 3; sind = 2;
A = reshape([A1 A2 A3], n, n, K);
#AH, Z = phess(A; h);
@time S, Z, eigs, ischur = pschur(A; sind);
@test check_psim(A, Z, S) && istriu(S[:,:,ischur],-1)

ev = eigvals(A[:,:,1]*A[:,:,2]*A[:,:,3])
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))


i1 = reverse(1:K)
@time a, e = psreduc_reg(view(A,:,:,i1)); eigs = eigvals(a,e);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

# modified MB03VD example
A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
A2 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.] .+ 1;
A3 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.] .- 1;
n = 4; K = 3; 
A = reshape([A1 A2 A3], n, n, K); 
M = PeriodicArray(A,K);
MA = PeriodicMatrix([A1,A2,A3],K); 

@time eigs = pseig(A); ev = pseig(A,fast = true);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)
@test sort(eigs, by = real) ≈ sort(ev, by = real)  
@test sort(eigs, by = abs) ≈ sort(ev, by = abs)  
@test sort(psceig(M),by = real) ≈ sort(psceig(MA),by = real)


@time eigs = pseig(A); ev = pseig(M,fast = true);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)
@test sort(eigs, by = real) ≈ sort(ev, by = real)  
@test sort(eigs, by = abs) ≈ sort(ev, by = abs)  



@time AH, Z, ihess = phess(A; rev = false);
@test check_psim(A,Z,AH; rev = false) && istriu(AH[:,:,ihess],-1) && ihess == 1

@time AH, Z, ihess = phess(A, hind = 2, rev = false);
@test check_psim(A,Z,AH; rev = false) && istriu(AH[:,:,ihess],-1) && ihess == 2

@time AH, Z, ihess = phess(A; rev = true); 
@test check_psim(A,Z,AH; rev = true) && istriu(AH[:,:,ihess],-1) && ihess == 1

@time AH, Z, ihess = phess(A; hind = 2, rev = true);
@test check_psim(A,Z,AH; rev = true) && istriu(AH[:,:,ihess],-1) && ihess == 2

ev = eigvals(A[:,:,1]*A[:,:,2]*A[:,:,3])
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
@test check_psim(A,Z,S; rev = false) && istriu(S[:,:,ischur],-1) && ischur == 1
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

@time S, Z, eigs, ischur, α, γ = pschur(A, sind = 2, rev = false); 
@test check_psim(A,Z,S; rev = false) && istriu(S[:,:,ischur],-1) && ischur == 2
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

ev = eigvals(A[:,:,3]*A[:,:,2]*A[:,:,1])
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = true); 
@test check_psim(A,Z,S; rev = true) && istriu(S[:,:,ischur],-1) && ischur == 1
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

@time S, Z, eigs, ischur, α, γ = pschur(A; sind = 2, rev = true); 
@test check_psim(A,Z,S; rev = true) && istriu(S[:,:,ischur],-1) && ischur == 2
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

ev = eigvals(A[:,:,3]*A[:,:,2]*A[:,:,1])
a, e = psreduc_reg(A); eigs = eigvals(a,e);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

A = [A1, A2, A3];
@time eigs = pseig(A); ev = pseig(A,fast = true);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

@time eigs = pseig(A; rev = false); ev = pseig(A; fast = true, rev = false);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

@time eigs = pseig(A; rev = false, istart = 2); 
@time ev = pseig(A; fast = true, rev = false, istart = 2);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)



A3 = rand(5,4); A2 = rand(4,3); A1 = rand(3,5); 
#A = [A1, A2, A3]; 
K = 3; 
A = PeriodicMatrix([A1, A2, A3],K);
@time eigs = pseig(A); ev = pseig(A,fast = true);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)
@test sort(eigs, by = real) ≈ sort(ev, by = real)  
@test sort(eigs, by = abs) ≈ sort(ev, by = abs)  


ev = eigvals(prod(A.M[K:-1:1])); nmin = minimum(size.(A.M,1))
@time S, Z, eigs, ischur, α, γ = pschur(A.M; rev = true);
@test check_psim(A.M,Z,S; rev = true) && istriu(S[ischur][1:nmin,1:nmin],-1) && eigs == α.*γ
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

A = [A3, A2, A1];
ev = eigvals(prod(A)); nmin = minimum(size.(A,1))

@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
@test check_psim(A,Z,S; rev = false) && istriu(S[ischur][1:nmin,1:nmin],-1) && eigs == α.*γ
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)


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



# constant dimensions
Ad = PeriodicMatrix([[1. 0; 0 0], [1 1;1 1], [0 1; 1 0]], 3);
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
for solver in ("non-stiff", "stiff", "linear", "symplectic", "noidea")
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
for solver in ("non-stiff", "stiff", "linear", "symplectic", "noidea")
      @time cvals = psceig(Afun, 500; solver, reltol = 1.e-10)
      @test cvals ≈ [2; -13]
end  

  
# #using BenchmarkTools
# solver = "non-stiff"
# at1(t) = [0 1; -10*cos(t) -24-10*sin(t)];
# Afun1=PeriodicFunctionMatrix(at1,2pi); 
# solvers = ("non-stiff", "stiff", "linear", "symplectic", "noidea")
# global k = 1
# for i = 1:5
#       println("solver = $(solvers[k])")
#       @btime  monodromy(Afun1, 500; solver = solvers[k], reltol = 1.e-10, abstol = 1.e-10);
#       k +=1
#       # @btime psceig(Afun, 500; solver, reltol = 1.e-10)
# end


# at1(t) = [0 1; -10*cos(t) -24-10*sin(t)];
# Afun1=PeriodicFunctionMatrix(at1,2pi); 
# println("solver = non-stiff")
# @btime  monodromy(Afun1, 500; solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10);
# println("solver = stiff")
# @btime  monodromy(Afun1, 500; solver = "stiff", reltol = 1.e-10, abstol = 1.e-10);
# println("solver = linear")
# @btime  monodromy(Afun1, 500; solver = "linear", reltol = 1.e-10, abstol = 1.e-10);
# println("solver = symplectic")
# @btime  monodromy(Afun1, 500; solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10);
# println("solver = noidea")
# @btime  monodromy(Afun1, 500; solver = "noidea", reltol = 1.e-10, abstol = 1.e-10);


# at(t) = [BigFloat(0) 1; -10*cos(t) -24-10*sin(t)];
# Afun=PeriodicFunctionMatrix(at,2pi); 

# f(u,p,t) = at(t)*u;
# u0 = Matrix{BigFloat}(I,2,2);
# tspan = (BigFloat(0.),2BigFloat(pi))
# prob = ODEProblem(f, u0, tspan)
# sol = solve(prob, Vern9(), rtol = 1.e-50, atol = 1.e-50, save_evrystep = false)
# cvals = log.(complex(eigvals(sol[end])))/2/pi

# at(t) = [BigFloat(0) 1; -10*cos(t) -24-10*sin(t)];
# f!(du,u,p,t) = mul!(du,at(t),u)
# u0 = Matrix{BigFloat}(I,2,2);
# tspan = (BigFloat(0.),2BigFloat(pi))
# prob1 = ODEProblem(f!, u0, tspan)
# sol1 = solve(prob1, Vern9(), rtol = 1.e-50, atol = 1.e-50, save_evrystep = false)
# cvals1 = log.(complex(eigvals(sol1[end])))/2/pi


# at(t) = [BigFloat(0) 1; -10*cos(t) -24-10*sin(t)];
# Afun=PeriodicFunctionMatrix(at,2BigFloat(pi)); 
# @time Φ = tvstm(Afun, Afun.period; reltol = 1.e-50, abstol = 1.e-50)  # running forever???
# cvals = log.(complex(eigvals(Φ)))/2/pi 
# println("cvals = $cvals  -- No one digit accuracy")
# @test maximum(abs.(eigvals(Φ))) > 1

# @variables x
# Cx(x::Real) = [ sin(x)+cos(2*x) 1]

# Anum = PeriodicSymbolicMatrix(Cx(x),2pi)

# # using the lifted periodic system
# ap = [A3,A2,A1]; ep = [E3,E2,E1]; bp = [rand(3,0), rand(3,0), rand(3,0)];
# cp = [rand(0,3), rand(0,3), rand(0,3)]; dp = [rand(0,0), rand(0,0), rand(0,0)];
# psys = PeriodicDiscreteDescriptorStateSpace{Float64}(ap,ep,bp,cp,dp,-1)
# poles = gpole(dss(psys))

# M, N = psreduc_fast([A3,A2,A1],[E3,E2,E1],atol = 1.e-7);
# poles = pzeros(M,N)[1]



# A1 = rand(3,4); A2 = rand(4,3); A = [A1,A2];
# E1 = rand(3,3); E2 = rand(4,4); E = [E1,E2];

# M, N = psreduc_fast(A,E,atol = 1.e-7)
# #E1 = eye(Float64,3); E2 = rand(4,4); E = [E1,E2];
# B = [rand(3,2), rand(4,1)];
# C = [rand(2,4), rand(3,3)];
# D = [rand(2,2), rand(3,1)];
# psys = PeriodicDiscreteDescriptorStateSpace{Float64}(A,E,B,C,D,-1)
# sys = dss(psys)

# ev = eigvals(inv(E2)*A2*inv(E1)*A1);
# poles = gpole(sys);

# @test sort(real(poles)) ≈ sort(real(ev)) && 
#       sort(imag(poles)) ≈ sort(imag(ev))

# sys = dss(psys, kstart = 2, compacted = true, atol = 1.e-7)

# ev = eigvals(inv(E1)*A1*inv(E2)*A2);
# poles = gpole(sys);

# @test sort(real(poles)) ≈ sort(real(ev)) && 
#       sort(imag(poles)) ≈ sort(imag(ev))


# psys = PeriodicDiscreteDescriptorStateSpace{Float64}(A,I,B,C,D,-1)
# sys = dss(psys)

# ev = eigvals(A2*A1);
# poles = gpole(sys);

# @test sort(real(poles)) ≈ sort(real(ev)) && 
#       sort(imag(poles)) ≈ sort(imag(ev))

# sys = dss(psys, kstart = 2, compacted = true, atol = 1.e-7)

# ev = eigvals(A1*A2);
# poles = gpole(sys);

# @test sort(real(poles)) ≈ sort(real(ev)) && 
#       sort(imag(poles)) ≈ sort(imag(ev))



end
end

