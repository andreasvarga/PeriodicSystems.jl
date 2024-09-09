module Test_psdric

using PeriodicSystems
using DescriptorSystems
using MatrixEquations
using Symbolics
using Test
using LinearAlgebra
#using JLD

println("Test_psdric")

@testset "prdric" begin

p = 10; n = 10; m = 5; period = 10;
A = PeriodicMatrix([rand(n,n) for i in 1:p],period);
B = PeriodicMatrix([rand(n,m) for i in 1:p],period);
R = PeriodicMatrix(eye(Float64,m),period; nperiod = period);
C = PeriodicMatrix([rand(m,n) for i in 1:p],period);
Q = C'*C; Q = (Q+Q')/2

@time X, EVALS, G = prdric(A,B,R,Q);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,R,Q,fast = false, nodeflate = false);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,R,Q,fast = false, nodeflate = true);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

                           
p = 5; na = [10, 8, 6, 4, 2]; ma = circshift(na,-1);
m = 5; period = 10;
A = PeriodicMatrix([rand(ma[i],na[i]) for i in 1:p],period);
B = PeriodicMatrix([rand(ma[i],m) for i in 1:p],period);
R = PeriodicMatrix(eye(Float64,m),period; nperiod = rationalize(A.period/A.Ts).num);
C = PeriodicMatrix([rand(m,na[i]) for i in 1:p],period);
Q = C'*C;  Q = (Q+Q')/2

@time X, EVALS, G = prdric(A,B,R,Q);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = length(EVALS)
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev[1:nc]))) < 1.e-6

@time X, EVALS, G = prdric(A,B,R,Q,fast = false,nodeflate = false);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = minimum(na)

@test norm(res) < 1.e-6 && norm(sort(real(EVALS[1:nc]))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS[1:nc]))-sort(imag(ev[1:nc]))) < 1.e-6                         


@time X, EVALS, G = prdric(A,B,R,Q,fast = false,nodeflate = true);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = minimum(na)

@test norm(res) < 1.e-6 && norm(sort(real(EVALS[1:nc]))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS[1:nc]))-sort(imag(ev[1:nc]))) < 1.e-6    
                           
                           
                           
p = 5; na = [2, 6, 4, 5, 8]; ma = circshift(na,-1);
m = 2; period = 10;
A = PeriodicMatrix([rand(ma[i],na[i]) for i in 1:p],period);
B = PeriodicMatrix([rand(ma[i],m) for i in 1:p],period);
R = PeriodicMatrix(eye(Float64,m),period; nperiod = rationalize(A.period/A.Ts).num);
C = PeriodicMatrix([rand(m,na[i]) for i in 1:p],period);
Q = C'*C;  Q = (Q+Q')/2

@time X, EVALS, G = prdric(A,B,R,Q; fast = true);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = minimum(na)
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev[1:nc]))) < 1.e-6

@time X, EVALS, G = prdric(A,B,R,Q,fast = false);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = minimum(na)

@test norm(res) < 1.e-6 && norm(sort(real(EVALS[1:nc]))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS[1:nc]))-sort(imag(ev[1:nc]))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,R,Q,fast = false,nodeflate = true);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
nc = minimum(na)

@test norm(res) < 1.e-6 && norm(sort(real(EVALS[1:nc]))-sort(real(ev[1:nc]))) < 1.e-6 &&
                           norm(sort(imag(EVALS[1:nc]))-sort(imag(ev[1:nc]))) < 1.e-6 


# Hench & Laub's IEEE TAC 1994: Example 2     
A1 =  [-3 2 9; 0 0 -4; 3 -2 3]; B1 = [1;1;0;;]; C1 = [1 0 0]; R1 = [1;;];             
A2 =  [6 -3 0; 4 -2 2; 2 -1 4]; B2 = [0;1;0;;]; C2 = [0 1 0]; R2 = [1;;];             
A3 =  [2 -3 -3; 4 -15 -3; -2 9 1]; B3 = [0;1;1;;]; C3 = [0 0 1]; R3 = [1;;];     
A = PeriodicMatrix([A1, A2, A3],10);
B = PeriodicMatrix([B1, B2, B3],10);
R = PeriodicMatrix([R1, R2, R3],10);
Q = PeriodicMatrix([C1'*C1, C2'*C2, C3'*C3],10);


@time X, EVALS, G = prdric(A,B,R,Q; fast = true);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 


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
@time psysa = psc2d(psysc,period/K,reltol = 1.e-10);
psys = convert(PeriodicStateSpace{PeriodicMatrix},psysa);

A = psys.A; B = psys.B; 
q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; r = [1.e-11;;];
Q = PeriodicMatrix(q, period; nperiod=K); R = PeriodicMatrix(r,period; nperiod=K);
Qa = PeriodicArray(q, period; nperiod=K); Ra = PeriodicArray(r,period; nperiod=K);

@time X, EVALS, G = prdric(A,B,R,Q,itmax = 2);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && issymmetric(X) && 
      norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
      norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,r,q; itmax = 2);  
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(r+B'*Xs*B)*B'*Xs)*A-q;
@test norm(res)/norm(X) < 1.e-6 && issymmetric(X) && 
      norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
      norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

A1 = A'; C1 = B';
@time X1, EVALS1, G1 = pfdric(A1,C1,R,Q; itmax = 2);  
ev1 = pseig(A1-G1*C1)
Xs1 = pmshift(X1)
res1 = Xs1-A1*(X1-X1*C1'*inv(R+C1*X1*C1')*C1*X1)*A1'-Q;
@test norm(res1)/norm(X1) < 1.e-6 && issymmetric(X1) && 
      norm(sort(real(EVALS1))-sort(real(ev1))) < 1.e-6 && 
      norm(sort(imag(EVALS1))-sort(imag(ev1))) < 1.e-6 

@time X1, EVALS1, G1 = pfdric(A1,C1,r,q; itmax = 2);  
Xs1 = pmshift(X1)
res1 = Xs1-A1*(X1-X1*C1'*inv(r+C1*X1*C1')*C1*X1)*A1'-q;
@test norm(res1)/norm(X1) < 1.e-6 


p = 10; n = 10; m = 5; period = 10;
A = PeriodicArray(rand(n,n,p),period)
B = PeriodicArray(rand(n,m,p),period)
C = PeriodicArray(rand(m,n,p),period)
R = PeriodicArray(eye(Float64,m),period; nperiod = period)
Q = C'*C;  Q = (Q+Q')/2
@time X, EVALS, G = prdric(A,B,R,Q);

ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 


# Hench & Laub's IEEE TAC 1994: Example 2     
A1 =  [-3 2 9; 0 0 -4; 3 -2 3]; B1 = [1;1;0;;]; C1 = [1 0 0]; R1 = [1;;];             
A2 =  [6 -3 0; 4 -2 2; 2 -1 4]; B2 = [0;1;0;;]; C2 = [0 1 0]; R2 = [1;;];             
A3 =  [2 -3 -3; 4 -15 -3; -2 9 1]; B3 = [0;1;1;;]; C3 = [0 0 1]; R3 = [1;;];     
A = PeriodicArray(reshape([A1 A2 A3],3,3,3),10);
B = PeriodicArray(reshape([B1 B2 B3],3,1,3),10);
R = PeriodicArray(reshape([R1 R2 R3],1,1,3),10);
Q = PeriodicArray(reshape([C1'*C1 C2'*C2 C3'*C3],3,3,3),10);


@time X, EVALS, G = prdric(A,B,R,Q; fast = true);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 


@time X, EVALS, G = prdric(A,B,R,Q; fast = false);  
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res)/norm(X) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6   
                           
@time X, EVALS, G = prdric(A,B,R,Q; fast = false, nodeflate = true);  
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res)/norm(X) < 1.e-6 && norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 &&
                           norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6   
                           
                           
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
psys = convert(PeriodicStateSpace{PeriodicArray},psc2d(psysc,period/K,reltol = 1.e-10));
A = convert(PeriodicArray,psys.A); B = convert(PeriodicArray,psys.B); 
q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; r = [1.e-11;;];
Q = PeriodicArray(q,period; nperiod=K); R = PeriodicArray(r,period; nperiod=K);

@time X, EVALS, G = prdric(A,B,R,Q,itmax = 2, fast = true);
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(R+B'*Xs*B)*B'*Xs)*A-Q;
@test norm(res) < 1.e-6 && issymmetric(X) && 
      norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
      norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,r,q; fast = false, itmax = 2);  
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(r+B'*Xs*B)*B'*Xs)*A-q;
@test norm(res)/norm(X) < 1.e-6 && issymmetric(X) && 
      norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
      norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

@time X, EVALS, G = prdric(A,B,r,q; fast = false, nodeflate = true, itmax = 2);  
ev = pseig(A-B*G)
Xs = pmshift(X)
res = X-A'*(Xs -Xs*B*inv(r+B'*Xs*B)*B'*Xs)*A-q;
@test norm(res)/norm(X) < 1.e-6 && issymmetric(X) && 
      norm(sort(real(EVALS))-sort(real(ev))) < 1.e-6 && 
      norm(sort(imag(EVALS))-sort(imag(ev))) < 1.e-6 

A1 = A'; C1 = B';
@time X1, EVALS1, G1 = pfdric(A1,C1,R,Q; fast = false, nodeflate = true, itmax = 2);  
ev1 = pseig(A1-G1*C1)
Xs1 = pmshift(X1)
res1 = Xs1-A1*(X1-X1*C1'*inv(R+C1*X1*C1')*C1*X1)*A1'-Q;
@test norm(res1)/norm(X1) < 1.e-6 && issymmetric(X1) && 
      norm(sort(real(EVALS1))-sort(real(ev1))) < 1.e-6 && 
      norm(sort(imag(EVALS1))-sort(imag(ev1))) < 1.e-6 

@time X1, EVALS1, G1 = pfdric(A1,C1,r,q; fast = false, nodeflate = true, itmax = 2);  
Xs1 = pmshift(X1)
res1 = Xs1-A1*(X1-X1*C1'*inv(r+C1*X1*C1')*C1*X1)*A1'-q;
@test norm(res1)/norm(X1) < 1.e-6 
end #prdare
end
