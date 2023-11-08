module Test_pschur

using PeriodicSystems
using Test
using LinearAlgebra
using LinearAlgebra: BlasInt
using MatrixPencils
#using BenchmarkTools

println("Test_psfutils")

@testset "test_psfutils" begin



# MB03BD example
A1 = Matrix{Float64}(I,3,3); A2 = [   1.0   2.0   0.0; 4.0  -1.0   3.0; 0.0   3.0   1.0]; A3 = Matrix{Float64}(I,3,3); 
E1 =  [2.0   0.0   1.0; 0.0  -2.0  -1.0; 0.0   0.0   3.0]; E2 = Matrix{Float64}(I,3,3); 
E3 = [ 1.0   0.0   1.0; 0.0   4.0  -1.0; 0.0   0.0  -2.0];
ev = eigvals(inv(E1)*A2*inv(E3))
ihess = 2

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

mb03bd!('T','C','I',QIND,3,3,ihess,1,3,S,A,Q,ALPHAR, ALPHAI, BETA, SCAL, LIWORK, LDWORK)

poles = (ALPHAR+im*ALPHAI) ./ BETA .* (2. .^SCAL)
println("poles = $poles")

@test sort(real(poles)) ≈ sort(real(ev)) && 
      sort(imag(poles)) ≈ sort(imag(ev))

ev1 = eigvals(inv(A[:,:,1])*A[:,:,2]*inv(A[:,:,3]))

Aw = reshape([A2 inv(E3) inv(E1)],3,3,3);
mb03wd!('S','I',3,3,1,3,1,3,Aw,Q,ALPHAR, ALPHAI, LDWORK)

poles = (ALPHAR+im*ALPHAI) 

@test sort(real(poles)) ≈ sort(real(ev)) && 
      sort(imag(poles)) ≈ sort(imag(ev))


k = 3; nc = 3; kschur = ihess; n = [3,3,3]; ni = [0,0,0]; s = S; select = [0,0,1]; 
t = zeros(0); [push!(t,A[:,:,k-i+1][:]...) for i in 1:k]
q = zeros(0); [push!(q,Q[:,:,k-i+1][:]...) for i in 1:k]
nn = nc*nc; ldt = nc*ones(Int,k); ixt = collect(1:nn:k*nn)
ldq = ldt; ixq = ixt;
tol = 20.; ldwork = max(42*k + max(nc,10), 80*k - 48) 

m, info = mb03kd!('U','S', k, nc, kschur, n, ni, s, select, t, ldt, ixt, q, ldq, ixq, tol, ldwork)
T = zeros(3,3,3); [T[:,:,k-i+1] = reshape(t[ixt[i]:ixt[i]+nn-1],nc,nc) for i in 1:k]
ev2 = eigvals(inv(T[:,:,1])*T[:,:,2]*inv(T[:,:,3]))
@test sort(real(poles)) ≈ sort(real(ev2)) && 
      sort(imag(poles)) ≈ sort(imag(ev2))

# MB03VD example
A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
A2 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.];
n = 4; K = 2; 
A = reshape([A1 A2], n, n, K);
@time H, Z, ihess = phess(A; rev = false);
@test check_psim(A,Z,H; rev = false) && istriu(H[:,:,ihess],-1) && ihess == 1

A1 = copy(A);
@time H, Z, ihess = phess!(A1; rev = false);
@test check_psim(A, Z, H; rev = false) && istriu(A1[:,:,ihess],-1) && ihess == 1


@time H1, Z1, ihess1 = phess(A; rev = false, withZ = false);
@test H[1] == H1[1] && H[2] == H1[2] 

@time H, Z, ihess = phess(A, hind = 2, rev = false);
@test check_psim(A,Z,H; rev = false) && istriu(H[:,:,ihess],-1) && ihess == 2

@time H1, Z1, ihess1 = phess(A, hind = 2, rev = false, withZ = false);
@test H[1] == H1[1] && H[2] == H1[2]  

@time H, Z, ihess = phess(A; rev = true); 
@test check_psim(A,Z,H; rev = true) && istriu(H[:,:,ihess],-1) && ihess == 1

A1 = copy(A);
@time H, Z, ihess = phess!(A1; rev = true);
@test check_psim(A, Z, H; rev = true) && istriu(A1[:,:,ihess],-1) && ihess == 1


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

A1 = copy(A);
@time S, Z, eigs, ischur, = pschur!(A1; rev = false);
@test check_psim(A, Z, S; rev = false) && istriu(A1[:,:,ischur],-1) && ischur == 1


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

A1 = copy(A);
@time S, Z, eigs, ischur, = pschur!(A1; rev = true);
@test check_psim(A, Z, S; rev = true) && istriu(A1[:,:,ischur],-1) && ischur == 1


@time S, Z, eigs, ischur, α, γ = pschur(A; sind = 2, rev = true); 
@test check_psim(A,Z,S; rev = true) && istriu(S[:,:,ischur],-1) && ischur == 2
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      sort(imag(eigs)) ≈ sort(imag(ev))

A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3; 1.  0.  2.  1.]; 
n = 4; K = 19; hind = 10; rev = false; 
A = reshape(hcat([rand()*A1 for i in 1:K]...), n, n, K);
@time H, Z, ihess = phess(A; hind, rev);
@test check_psim(A,Z,H; rev) && istriu(H[:,:,ihess],-1) && ihess == hind

@time H, Z, ihess = phess1(A; hind, rev);
@test check_psim(A,Z,H; rev) && istriu(H[:,:,ihess],-1) && ihess == hind

@time S, Z, eigs, ischur, = pschur(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)

@time S, Z, eigs, ischur, = pschur1(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)

# this would fail
# n =4; K = 19; hind = 10; rev = false; 
# A = reshape(hcat([rand()*A1 for i in 1:K]...), n, n, K);
# @time S, Z, eigs, ischur, = pschur2(A; sind = hind, rev);
# @test check_psim(A,Z,S; rev)

n =4; K = 15; hind = 10; rev = false; 
A = reshape(hcat([rand()*A1 for i in 1:K]...), n, n, K);
@time S, Z, eigs, ischur, = pschur2(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)

hind = 14; rev = true; 
@time H, Z, ihess = phess(A; hind, rev);
@test check_psim(A,Z,H; rev) && istriu(H[:,:,ihess],-1) && ihess == hind

@time H, Z, ihess = phess1(A; hind, rev);
@test check_psim(A,Z,H; rev) && istriu(H[:,:,ihess],-1) && ihess == hind

@time S, Z, eigs, ischur, = pschur(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)

@time S, Z, eigs, ischur, = pschur1(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)

# this would fail
# n =4; K = 19; hind = 10; rev = false; 
# A = reshape(hcat([rand()*A1 for i in 1:K]...), n, n, K);
# @time S, Z, eigs, ischur, = pschur2(A; sind = hind, rev);
# @test check_psim(A,Z,S; rev)

n =4; K = 15; hind = 10; rev = false; 
A = reshape(hcat([rand()*A1 for i in 1:K]...), n, n, K);
@time S, Z, eigs, ischur, = pschur2(A; sind = hind, rev);
@test check_psim(A,Z,S; rev)


A1 = [1.5 -.7 3.5 -.7; 1.  0.  2.  3.; 1.5 -.7 2.5 -.3]; 
A2 = [1.5 -.7 3.5; 1.  0.  2.; 1.5 -.7 2.5 ; 1.  0.  2.];
Ai=[A1,A2]; 
A = [Ai[j] for j = 1:2,  i in 1:9][:]

rev = false; 
@time S, Z, eigs, ischur, = pschur(A; rev);
@test check_psim(A,Z,S; rev)

rev = true; 
@time S, Z, eigs, ischur, = pschur(A; rev);
@test check_psim(A,Z,S; rev)

# @time S1, Z1, eigs1, ischur1 = pschurw(A; rev = false);
# @test check_psim(A,Z1,S1; rev = false)


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


# i1 = reverse(1:K)
# @time a, e = psreduc_reg(view(A,:,:,i1)); eigs = eigvals(a,e);
# @test sort(real(eigs)) ≈ sort(real(ev)) && 
#       sort(imag(eigs)) ≈ sort(imag(ev))

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

# ev = eigvals(A[:,:,3]*A[:,:,2]*A[:,:,1])
# a, e = psreduc_reg(A); eigs = eigvals(a,e);
# @test sort(real(eigs)) ≈ sort(real(ev)) && 
#       sort(imag(eigs)) ≈ sort(imag(ev))

AA = [A[:,:,3], A[:,:,2], A[:,:,1]];
@time S, Z, eigs, ischur, α, γ = pschur(AA; rev = false);
select =  real(eigs) .< 0
@time psordschur!(S, Z, select; schurindex = ischur, rev = false)  
@test check_psim(AA, Z, S; rev = false)  
MatrixPencils.ordeigvals(S[1]*S[2]*S[3]) 

AA = [A[:,:,1], A[:,:,2], A[:,:,3]];
@time S, Z, eigs, ischur, α, γ = pschur(AA; rev = true);
select =  real(eigs) .< 0
@time psordschur!(S, Z, select; schurindex=ischur, rev = true)  
@test check_psim(AA,Z,S; rev = true)  
MatrixPencils.ordeigvals(S[3]*S[2]*S[1])  

AA = [A[:,:,3], A[:,:,2], A[:,:,1]];
@time S, Z, eigs, ischur, α, γ = pschur(AA; rev = false);
select =  real(eigs) .< 0
@time psordschur!(S, Z, select; schurindex = ischur, rev = false)  
@test check_psim(AA, Z, S; rev = false)  
MatrixPencils.ordeigvals(S[1]*S[2]*S[3]) 

@time S, Z, eigs, ischur, α, γ = pschur(A; rev = true);
select =  real(eigs) .< 0
@time psordschur!(S, Z, select; schurindex=ischur, rev = true)  
@test check_psim(A,Z,S; rev = true)  
MatrixPencils.ordeigvals(S[:,:,3]*S[:,:,2]*S[:,:,1])  

@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
select =  real(eigs) .< 0
@time psordschur!(S, Z, select; schurindex = ischur, rev = false)  
@test check_psim(A, Z, S; rev = false)  
MatrixPencils.ordeigvals(S[:,:,1]*S[:,:,2]*S[:,:,3])  


A = [A1, A2, A3];
@time eigs = pseig(A); ev = pseig(A,fast = true);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

@time eigs = pseig(A; rev = false); ev = pseig(A; fast = true, rev = false);
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

@time eigs = pseig(A, 2; rev = false); 
@time ev = pseig(A, 2; fast = true, rev = false);
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
@test isapprox(sort(real(eigs)),sort(real(ev)), atol = 1.e-7) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

A = [A3, A2, A1];
ev = eigvals(prod(A)); nmin = minimum(size.(A,1))

@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
@test check_psim(A,Z,S; rev = false) && istriu(S[ischur][1:nmin,1:nmin],-1) && eigs == α.*γ
@test sort(real(eigs)) ≈ sort(real(ev)) && 
      isapprox(sort(imag(eigs)),sort(imag(ev)), atol = 1.e-7)

A = [A3, A2, A1];
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = false);
select =  (real(eigs) .< 0)[1:3]
@time psordschur!(S, Z, select; schurindex = ischur, rev = false)  
@test check_psim(A, Z, S; rev = false)  
MatrixPencils.ordeigvals(S[1]*S[2]*S[3]) 

A = [A1, A2, A3];
@time S, Z, eigs, ischur, α, γ = pschur(A; rev = true);
select =  (real(eigs) .< 0)[1:3]
@time psordschur!(S,Z,select; schurindex=ischur, rev = true)  
@test check_psim(A,Z,S; rev = true)  
MatrixPencils.ordeigvals(S[3]*S[2]*S[1])  


# Example Hench & Laub IEETAC 1994
A1 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1; 2 1 0 -2 0; -4 -2 4 -4 0];
E1 = [4 5 4 -3 -5; 4 -3 0 5 -5; 0 0 0 -4 3; -4 -2 3 1 2; 0 0 -1 1 -2];
A2 = [-2 -1 2 5 0; -1 -2 2 0 1; -1 -5 0 5 -1; 0 -1 2 -2 1; 0 -2 2 0 5];
E2 = [0 2 1 5 -2; 2 3 -3 5 3; 2 -4 -3 -1 3; -3 -5 -5 -2 0; 4 3 -1 2 -2];
A3 = [-4 3 5 2 2; 2 -5 -2 0 2; -1 -2 5 3 0; 0 1 -3 -5 0; 0 0 4 -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];

ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1);
As, Es, Q, Z, evp, sind, α, γ = PeriodicSystems.pschur([A1, A2, A3], [E1, E2, E3]; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
@test Q[1]'*A1*Z[1] ≈ As[1] && Q[2]'*A2*Z[2] ≈ As[2] && Q[3]'*A3*Z[3] ≈ As[3] &&
      Q[1]'*E1*Z[2] ≈ Es[1] && Q[2]'*E2*Z[3] ≈ Es[2] && Q[3]'*E3*Z[1] ≈ Es[3]


ev1 = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))
As, Es, Q1, Z1, evp1, sind1, α1, γ1 = PeriodicSystems.pschur([A1, A2, A3], [E1, E2, E3]; rev=false); evp1
@test sort(real(ev1)) ≈ sort(real(evp1)) && sort(imag(ev1)) ≈ sort(imag(evp1))
@test Q1[1]'*A1*Z1[1] ≈ As[1] && Q1[2]'*A2*Z1[2] ≈ As[2] && Q1[3]'*A3*Z1[3] ≈ As[3] &&
      Q1[2]'*E1*Z1[1] ≈ Es[1] && Q1[3]'*E2*Z1[2] ≈ Es[2] && Q1[1]'*E3*Z1[3] ≈ Es[3]


A1 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1];
E1 = [4 5 4 ; 4 -3 0 ; -4 -2 3 ];
A2 = [-2 -1 2; -1 -2 2];
E2 = [5 -2; 2 3];
A3 = [-4 3 ; 2 -5 ; -1 -2 ; 0 1 ; -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];

ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1)
As, Es, Q, Z, evp, sind, α, γ = PeriodicSystems.pschur([A1, A2, A3], [E1, E2, E3]; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && norm(sort(imag(ev)) - sort(imag(evp))) < 1.e-7
@test Q[1]'*A1*Z[1] ≈ As[1] && Q[2]'*A2*Z[2] ≈ As[2] && Q[3]'*A3*Z[3] ≈ As[3] &&
      Q[1]'*E1*Z[2] ≈ Es[1] && Q[2]'*E2*Z[3] ≈ Es[2] && Q[3]'*E3*Z[1] ≈ Es[3]


A1 = [-4 3 ; 2 -5 ; -1 -2 ; 0 1 ; -4 3];
E1 = [5 -2; 2 3];
A2 = [-2 -1 2; -1 -2 2];
E2 = [4 5 4 ; 4 -3 0 ; -4 -2 3 ];
A3 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];

ev = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))

As, Es, Q, Z, evp, sind, α, γ = PeriodicSystems.pschur([A1, A2, A3], [E1, E2, E3]; rev=false); evp
@test sort(real(ev)) ≈ sort(real(evp)) && norm(sort(imag(ev)) - sort(imag(evp))) < 1.e-7
@test Q[1]'*A1*Z[1] ≈ As[1] && Q[2]'*A2*Z[2] ≈ As[2] && Q[3]'*A3*Z[3] ≈ As[3] &&
      Q[2]'*E1*Z[1] ≈ Es[1] && Q[3]'*E2*Z[2] ≈ Es[2] && Q[1]'*E3*Z[3] ≈ Es[3]

A1 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1; 2 1 0 -2 0; -4 -2 4 -4 0];
E1 = [4 5 4 -3 -5; 4 -3 0 5 -5; 0 0 0 -4 3; -4 -2 3 1 2; 0 0 -1 1 -2];
A2 = [-2 -1 2 5 0; -1 -2 2 0 1; -1 -5 0 5 -1; 0 -1 2 -2 1; 0 -2 2 0 5];
E2 = [0 2 1 5 -2; 2 3 -3 5 3; 2 -4 -3 -1 3; -3 -5 -5 -2 0; 4 3 -1 2 -2];
A3 = [-4 3 5 2 2; 2 -5 -2 0 2; -1 -2 5 3 0; 0 1 -3 -5 0; 0 0 4 -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];
A = reshape([A1 A2 A3],5,5,3); E = reshape([E1 E2 E3],5,5,3);


ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1);
As, Es, Q, Z, evp, sind, α, γ = PeriodicSystems.pschur(A, E; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
@test Q[:,:,1]'*A1*Z[:,:,1] ≈ As[:,:,1] && Q[:,:,2]'*A2*Z[:,:,2] ≈ As[:,:,2] && Q[:,:,3]'*A3*Z[:,:,3] ≈ As[:,:,3] &&
      Q[:,:,1]'*E1*Z[:,:,2] ≈ Es[:,:,1] && Q[:,:,2]'*E2*Z[:,:,3] ≈ Es[:,:,2] && Q[:,:,3]'*E3*Z[:,:,1] ≈ Es[:,:,3]


ev1 = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))
As, Es, Q1, Z1, evp1, sind1, α1, γ1 = PeriodicSystems.pschur(A, E; rev=false); evp1
@test sort(real(ev1)) ≈ sort(real(evp1)) && sort(imag(ev1)) ≈ sort(imag(evp1))
@test Q1[:,:,1]'*A1*Z1[:,:,1] ≈ As[:,:,1] && Q1[:,:,2]'*A2*Z1[:,:,2] ≈ As[:,:,2] && Q1[:,:,3]'*A3*Z1[:,:,3] ≈ As[:,:,3] &&
      Q1[:,:,2]'*E1*Z1[:,:,1] ≈ Es[:,:,1] && Q1[:,:,3]'*E2*Z1[:,:,2] ≈ Es[:,:,2] && Q1[:,:,1]'*E3*Z1[:,:,3] ≈ Es[:,:,3]


A1 = [1. 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1; 2 1 0 -2 0; -4 -2 4 -4 0];
E1 = [4 5 4 -3 -5; 4 -3 0 5 -5; 0 0 0 -4 3; -4 -2 3 1 2; 0 0 -1 1 -2];
A2 = [-2 -1 2 5 0; -1 -2 2 0 1; -1 -5 0 5 -1; 0 -1 2 -2 1; 0 -2 2 0 5];
E2 = [0 2 1 5 -2; 2 3 -3 5 3; 2 -4 -3 -1 3; -3 -5 -5 -2 0; 4 3 -1 2 -2];
A3 = [-4 3 5 2 2; 2 -5 -2 0 2; -1 -2 5 3 0; 0 1 -3 -5 0; 0 0 4 -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];
A = reshape([A1 E1 A2 E2 A3 E3],5,5,6);

ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1);
As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur!(reshape([A1 E1 A2 E2 A3 E3],5,5,6), [trues(3) falses(3)]'[:]; rev=true, schurindex = 2); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
Q = view(Qt,:,:,2:2:6); Z = view(Qt,:,:,1:2:6);
@test Q[:,:,1]'*A1*Z[:,:,1] ≈ As[:,:,1] && Q[:,:,1]'*E1*Z[:,:,2] ≈ As[:,:,2] && 
      Q[:,:,2]'*A2*Z[:,:,2] ≈ As[:,:,3] && Q[:,:,2]'*E2*Z[:,:,3] ≈ As[:,:,4] &&
      Q[:,:,3]'*A3*Z[:,:,3] ≈ As[:,:,5] && Q[:,:,3]'*E3*Z[:,:,1] ≈ As[:,:,6] 
@test Qt[:,:,2]'*A[:,:,1]*Qt[:,:,1] ≈ As[:,:,1] && Qt[:,:,2]'*A[:,:,2]*Qt[:,:,3] ≈ As[:,:,2] && 
      Qt[:,:,4]'*A[:,:,3]*Qt[:,:,3] ≈ As[:,:,3] && Qt[:,:,4]'*A[:,:,4]*Qt[:,:,5] ≈ As[:,:,4] && 
      Qt[:,:,6]'*A[:,:,5]*Qt[:,:,5] ≈ As[:,:,5] && Qt[:,:,6]'*A[:,:,6]*Qt[:,:,1] ≈ As[:,:,6]

As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur(A, [trues(3) falses(3)]'[:]; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
@test Qt[:,:,2]'*A[:,:,1]*Qt[:,:,1] ≈ As[:,:,1] && Qt[:,:,2]'*A[:,:,2]*Qt[:,:,3] ≈ As[:,:,2] && 
      Qt[:,:,4]'*A[:,:,3]*Qt[:,:,3] ≈ As[:,:,3] && Qt[:,:,4]'*A[:,:,4]*Qt[:,:,5] ≈ As[:,:,4] && 
      Qt[:,:,6]'*A[:,:,5]*Qt[:,:,5] ≈ As[:,:,5] && Qt[:,:,6]'*A[:,:,6]*Qt[:,:,1] ≈ As[:,:,6]

select = real(evp) .< 0
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=true, schurindex = sind); 
@test Qt[:,:,2]'*A[:,:,1]*Qt[:,:,1] ≈ As[:,:,1] && Qt[:,:,2]'*A[:,:,2]*Qt[:,:,3] ≈ As[:,:,2] && 
      Qt[:,:,4]'*A[:,:,3]*Qt[:,:,3] ≈ As[:,:,3] && Qt[:,:,4]'*A[:,:,4]*Qt[:,:,5] ≈ As[:,:,4] && 
      Qt[:,:,6]'*A[:,:,5]*Qt[:,:,5] ≈ As[:,:,5] && Qt[:,:,6]'*A[:,:,6]*Qt[:,:,1] ≈ As[:,:,6]
evp2 = eigvals(inv(As[:,:,6])*As[:,:,5]*inv(As[:,:,4])*As[:,:,3]*inv(As[:,:,2])*As[:,:,1])
@test all((real(evp2) .< 0)[1:count(select)])

ev1 = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))

As, Qt, evp1, sind1, α1, γ1 = PeriodicSystems.pgschur!(reshape([A1 E1 A2 E2 A3 E3],5,5,6), [trues(3) falses(3)]'[:]; rev=false); evp1
@test sort(real(ev1)) ≈ sort(real(evp1)) && sort(imag(ev1)) ≈ sort(imag(evp1))
Q = view(Qt,:,:,1:2:6); Z = view(Qt,:,:,2:2:6);
@test Q[:,:,1]'*A1*Z[:,:,1] ≈ As[:,:,1] && Q[:,:,2]'*E1*Z[:,:,1] ≈ As[:,:,2] && 
      Q[:,:,2]'*A2*Z[:,:,2] ≈ As[:,:,3] && Q[:,:,3]'*E2*Z[:,:,2] ≈ As[:,:,4] && 
      Q[:,:,3]'*A3*Z[:,:,3] ≈ As[:,:,5] && Q[:,:,1]'*E3*Z[:,:,3] ≈ As[:,:,6]
@test Qt[:,:,1]'*A[:,:,1]*Qt[:,:,2] ≈ As[:,:,1] && Qt[:,:,3]'*A[:,:,2]*Qt[:,:,2] ≈ As[:,:,2] && 
      Qt[:,:,3]'*A[:,:,3]*Qt[:,:,4] ≈ As[:,:,3] && Qt[:,:,5]'*A[:,:,4]*Qt[:,:,4] ≈ As[:,:,4] && 
      Qt[:,:,5]'*A[:,:,5]*Qt[:,:,6] ≈ As[:,:,5] && Qt[:,:,1]'*A[:,:,6]*Qt[:,:,6] ≈ As[:,:,6]

select = abs.(evp1) .< 1
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=false, schurindex = sind1); 
@test Qt[:,:,1]'*A[:,:,1]*Qt[:,:,2] ≈ As[:,:,1] && Qt[:,:,3]'*A[:,:,2]*Qt[:,:,2] ≈ As[:,:,2] && 
      Qt[:,:,3]'*A[:,:,3]*Qt[:,:,4] ≈ As[:,:,3] && Qt[:,:,5]'*A[:,:,4]*Qt[:,:,4] ≈ As[:,:,4] && 
      Qt[:,:,5]'*A[:,:,5]*Qt[:,:,6] ≈ As[:,:,5] && Qt[:,:,1]'*A[:,:,6]*Qt[:,:,6] ≈ As[:,:,6]
evp2 = ordeigvals(As[:,:,1]*inv(As[:,:,2])*As[:,:,3]*inv(As[:,:,4])*As[:,:,5]*inv(As[:,:,6]))
@test all((abs.(evp2) .< 1)[1:count(select)])


A1 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1; 2 1 0 -2 0; -4 -2 4 -4 0];
E1 = [4 5 4 -3 -5; 4 -3 0 5 -5; 0 0 0 -4 3; -4 -2 3 1 2; 0 0 -1 1 -2];
A2 = [-2 -1 2 5 0; -1 -2 2 0 1; -1 -5 0 5 -1; 0 -1 2 -2 1; 0 -2 2 0 5];
E2 = [0 2 1 5 -2; 2 3 -3 5 3; 2 -4 -3 -1 3; -3 -5 -5 -2 0; 4 3 -1 2 -2];
A3 = [-4 3 5 2 2; 2 -5 -2 0 2; -1 -2 5 3 0; 0 1 -3 -5 0; 0 0 4 -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];
A = [A1, E1, A2, E2, A3, E3];

ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1);
As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur(A, [trues(3) falses(3)]'[:]; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
@test Qt[2]'*A[1]*Qt[1] ≈ As[1] && Qt[2]'*A[2]*Qt[3] ≈ As[2] && 
      Qt[4]'*A[3]*Qt[3] ≈ As[3] && Qt[4]'*A[4]*Qt[5] ≈ As[4] && 
      Qt[6]'*A[5]*Qt[5] ≈ As[5] && Qt[6]'*A[6]*Qt[1] ≈ As[6]

Q = view(Qt,2:2:6); Z = view(Qt,1:2:6);
@test Q[1]'*A1*Z[1] ≈ As[1] && Q[2]'*A2*Z[2] ≈ As[3] && Q[3]'*A3*Z[3] ≈ As[5] &&
      Q[1]'*E1*Z[2] ≈ As[2] && Q[2]'*E2*Z[3] ≈ As[4] && Q[3]'*E3*Z[1] ≈ As[6]

select = real(evp) .< 0
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=true, schurindex = sind); 
@test Qt[2]'*A[1]*Qt[1] ≈ As[1] && Qt[2]'*A[2]*Qt[3] ≈ As[2] && 
      Qt[4]'*A[3]*Qt[3] ≈ As[3] && Qt[4]'*A[4]*Qt[5] ≈ As[4] && 
      Qt[6]'*A[5]*Qt[5] ≈ As[5] && Qt[6]'*A[6]*Qt[1] ≈ As[6]
evp2 = ordeigvals(inv(As[6])*As[5]*inv(As[4])*As[3]*inv(As[2])*As[1])
@test all((real(evp2) .< 0)[1:count(select)])



ev = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))
As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur(A,[trues(3) falses(3)]'[:]; rev=false); evp
@test sort(real(ev)) ≈ sort(real(evp)) && sort(imag(ev)) ≈ sort(imag(evp))
@test Qt[1]'*A[1]*Qt[2] ≈ As[1] && Qt[3]'*A[2]*Qt[2] ≈ As[2] && 
      Qt[3]'*A[3]*Qt[4] ≈ As[3] && Qt[5]'*A[4]*Qt[4] ≈ As[4] && 
      Qt[5]'*A[5]*Qt[6] ≈ As[5] && Qt[1]'*A[6]*Qt[6] ≈ As[6]

Q1 = view(Qt,1:2:6); Z1 = view(Qt,2:2:6);
@test Q1[1]'*A1*Z1[1] ≈ As[1] && Q1[2]'*A2*Z1[2] ≈ As[3] && Q1[3]'*A3*Z1[3] ≈ As[5] &&
      Q1[2]'*E1*Z1[1] ≈ As[2] && Q1[3]'*E2*Z1[2] ≈ As[4] && Q1[1]'*E3*Z1[3] ≈ As[6]

select = abs.(evp) .< 1
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=false, schurindex = sind); 
@test Qt[1]'*A[1]*Qt[2] ≈ As[1] && Qt[3]'*A[2]*Qt[2] ≈ As[2] && 
      Qt[3]'*A[3]*Qt[4] ≈ As[3] && Qt[5]'*A[4]*Qt[4] ≈ As[4] && 
      Qt[5]'*A[5]*Qt[6] ≈ As[5] && Qt[1]'*A[6]*Qt[6] ≈ As[6]
evp2 = ordeigvals(As[1]*inv(As[2])*As[3]*inv(As[4])*As[5]*inv(As[6]))
@test all((abs.(evp2) .< 1)[1:count(select)])



A1 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1];
E1 = [4 5 4 ; 4 -3 0 ; -4 -2 3 ];
A2 = [-2 -1 2; -1 -2 2];
E2 = [5 -2; 2 3];
A3 = [-4 3 ; 2 -5 ; -1 -2 ; 0 1 ; -4 3];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];
A = [A1, E1, A2, E2, A3, E3];
nc = 2; 


ev = eigvals(inv(E3)*A3*inv(E2)*A2*inv(E1)*A1)
As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur(A,  [trues(3) falses(3)]'[:]; rev=true); evp
@test sort(real(ev)) ≈ sort(real(evp)) && norm(sort(imag(ev)) - sort(imag(evp))) < 1.e-7
@test Qt[2]'*A[1]*Qt[1] ≈ As[1] && Qt[2]'*A[2]*Qt[3] ≈ As[2] && 
      Qt[4]'*A[3]*Qt[3] ≈ As[3] && Qt[4]'*A[4]*Qt[5] ≈ As[4] && 
      Qt[6]'*A[5]*Qt[5] ≈ As[5] && Qt[6]'*A[6]*Qt[1] ≈ As[6]

Q = view(Qt,2:2:6); Z = view(Qt,1:2:6);
@test Q[1]'*A1*Z[1] ≈ As[1] && Q[2]'*A2*Z[2] ≈ As[3] && Q[3]'*A3*Z[3] ≈ As[5] &&
      Q[1]'*E1*Z[2] ≈ As[2] && Q[2]'*E2*Z[3] ≈ As[4] && Q[3]'*E3*Z[1] ≈ As[6]

select = real(evp[1:nc]) .> 0
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=true, schurindex = sind); 
@test Qt[2]'*A[1]*Qt[1] ≈ As[1] && Qt[2]'*A[2]*Qt[3] ≈ As[2] && 
      Qt[4]'*A[3]*Qt[3] ≈ As[3] && Qt[4]'*A[4]*Qt[5] ≈ As[4] && 
      Qt[6]'*A[5]*Qt[5] ≈ As[5] && Qt[6]'*A[6]*Qt[1] ≈ As[6]
evp2 = ordeigvals(inv(As[6])*As[5]*inv(As[4])*As[3]*inv(As[2])*As[1])
@test all((real(evp2[1:nc]) .> 0)[1:count(select)])



A1 = [-4 3 ; 2 -5 ; -1 -2 ; 0 1 ; -4 3];
E1 = [5 -2; 2 3];
A2 = [-2 -1 2; -1 -2 2];
E2 = [4 5 4 ; 4 -3 0 ; -4 -2 3 ];
A3 = [1 3 2 3 0; -3 0 -4 0 -2; -2 0 -4 2 1];
E3 = [-3 3 3 -4 2; -5 -4 -4 -3 3; -3 -1 5 1 2; 5 1 -2 2 1; -3 -3 -3 3 -5];
A = [A1, E1, A2, E2, A3, E3];
nc = 2; 

ev = eigvals(A1*inv(E1)*A2*inv(E2)*A3*inv(E3))

As, Qt, evp, sind, α, γ = PeriodicSystems.pgschur(A,  [trues(3) falses(3)]'[:]; rev=false); evp
@test sort(real(ev)) ≈ sort(real(evp)) && norm(sort(imag(ev)) - sort(imag(evp))) < 1.e-7

@test Qt[1]'*A[1]*Qt[2] ≈ As[1] && Qt[3]'*A[2]*Qt[2] ≈ As[2] && 
      Qt[3]'*A[3]*Qt[4] ≈ As[3] && Qt[5]'*A[4]*Qt[4] ≈ As[4] && 
      Qt[5]'*A[5]*Qt[6] ≈ As[5] && Qt[1]'*A[6]*Qt[6] ≈ As[6]

Q1 = view(Qt,1:2:6); Z1 = view(Qt,2:2:6);
@test Q1[1]'*A1*Z1[1] ≈ As[1] && Q1[2]'*A2*Z1[2] ≈ As[3] && Q1[3]'*A3*Z1[3] ≈ As[5] &&
      Q1[2]'*E1*Z1[1] ≈ As[2] && Q1[3]'*E2*Z1[2] ≈ As[4] && Q1[1]'*E3*Z1[3] ≈ As[6]

select = abs.(evp[1:nc]) .< 1
PeriodicSystems.pgordschur!(As, [trues(3) falses(3)]'[:], Qt, select; rev=false, schurindex = sind); 
@test Qt[1]'*A[1]*Qt[2] ≈ As[1] && Qt[3]'*A[2]*Qt[2] ≈ As[2] && 
      Qt[3]'*A[3]*Qt[4] ≈ As[3] && Qt[5]'*A[4]*Qt[4] ≈ As[4] && 
      Qt[5]'*A[5]*Qt[6] ≈ As[5] && Qt[1]'*A[6]*Qt[6] ≈ As[6]
evp2 = ordeigvals(As[1]*inv(As[2])*As[3]*inv(As[4])*As[5]*inv(As[6]))
@test all((abs.(evp2[1:nc]) .< 1)[1:count(select)])


end
end

