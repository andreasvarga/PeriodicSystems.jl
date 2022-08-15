module Test_pmops

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using ApproxFun
using Symbolics

println("Test_pmops")

@testset "pmops" begin

# generate periodic function matrices with same period and nperiod = 1
A(t) = [0  1; -10*cos(t)-1 -24-19*sin(t)]
X(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
Xder(t) = [-sin(t) 0; 0 cos(t)]  # derivative of the desired solution
C(t) = [ -sin(t)  -1-sin(t)-(-1-10cos(t))*(1+cos(t));
-1-sin(t)-(-1-10cos(t))*(1+cos(t))   cos(t)- 2(-24 - 19sin(t))*(1 + sin(t)) ]  # corresponding C
Cd(t) = [ sin(t)  -1-cos(t)-(-1-10cos(t))*(1+sin(t));
-1-cos(t)-(-1-10cos(t))*(1+sin(t))    -cos(t)-2(-24-19sin(t))*(1 + sin(t)) ] # corresponding Cd
A1(t) = [0  1; -0.5*cos(t)-1 -24-19*sin(t)]  # invertible matrix
A1inv(t) = [(24 + 19sin(t)) / (-1 - 0.5cos(t))  1 / (-1 - 0.5cos(t)); 1.0  0.0]  # invertible matrix


# PeriodicFunctionMatrix
At = PeriodicFunctionMatrix(A,2*pi)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,2*pi)
Xdert = PeriodicFunctionMatrix(Xder,2*pi)
@test At*Xt+Xt*At'+Ct ≈  derivative(Xt) ≈ Xdert
@test At'*Xt+Xt*At+Cdt ≈ -derivative(Xt)
@test norm(At*Xt+Xt*At'+Ct-derivative(Xt)) < 1.e-7
@test norm(At'*Xt+Xt*At+Cdt+derivative(Xt)) < 1.e-7


At = PeriodicFunctionMatrix(A,4*pi,nperiod=2)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,8*pi,nperiod=4)
Xdert = PeriodicFunctionMatrix(Xder,16*pi,nperiod=8)
@test At*Xt+Xt*At'+Ct ≈  derivative(Xt) ≈ Xdert
@test At'*Xt+Xt*At+Cdt ≈ -derivative(Xt)
@test norm(At*Xt+Xt*At'+Ct-derivative(Xt)) < 1.e-7
@test norm(At'*Xt+Xt*At+Cdt+derivative(Xt)) < 1.e-7

D = rand(2,2)
@test At+I == I+At && At*5 == 5*At && At*D ≈ -At*(-D) && iszero(At-At) && !iszero(At)
@test inv(At)*At ≈ I ≈ At*inv(At) && At+I == I+At
@test PeriodicFunctionMatrix(D,2*pi) == PeriodicFunctionMatrix(D,4*pi) && 
      PeriodicFunctionMatrix(D,2*pi) ≈ PeriodicFunctionMatrix(D,4*pi)


# HarmonicArray
Ah = convert(HarmonicArray,PeriodicFunctionMatrix(A,2*pi));
Ah1 = convert(HarmonicArray,PeriodicFunctionMatrix(A1,2*pi));
Ch = convert(HarmonicArray,PeriodicFunctionMatrix(C,2*pi));
Cdh = convert(HarmonicArray,PeriodicFunctionMatrix(Cd,2*pi));
Xh = convert(HarmonicArray,PeriodicFunctionMatrix(X,2*pi));
Xderh = convert(HarmonicArray,PeriodicFunctionMatrix(Xder,2*pi));
@test issymmetric(Ch) && issymmetric(Cdh) && issymmetric(Xh) && issymmetric(Xderh)
@test Ah*Xh+Xh*Ah'+Ch ≈  derivative(Xh) ≈ Xderh
@test Ah'*Xh+Xh*Ah+Cdh ≈ -derivative(Xh) 
@test norm(Ah*Xh+Xh*Ah'+Ch-derivative(Xh),Inf) < 1.e-7 && norm(derivative(Xh)- Xderh) < 1.e-7
@test norm(Ah'*Xh+Xh*Ah+Cdh+derivative(Xh),1) < 1.e-7

D = rand(2,2)
@test Ah+I == I+Ah && Ah*5 == 5*Ah && Ah*D ≈ -Ah*(-D) && iszero(Ah-Ah) && !iszero(Ah)
@test HarmonicArray(D,2*pi) == HarmonicArray(D,4*pi) && 
      HarmonicArray(D,2*pi) ≈ HarmonicArray(D,4*pi)
Ah1i = inv(Ah1)
@test Ah1i*Ah1 ≈ I ≈ Ah1*Ah1i 
@test hrchop(Ah1i; tol = 1.e-10) ≈ hrchop(Ah1i; tol = eps()) ≈ hrchop(Ah1i; tol = 1.e-10) 
@test hrtrunc(Ah1i) ≈ hrtrunc(Ah1i,10)

Ah = convert(HarmonicArray,PeriodicFunctionMatrix(A,4*pi));
Ch = convert(HarmonicArray,PeriodicFunctionMatrix(C,2*pi));
Cdh = convert(HarmonicArray,PeriodicFunctionMatrix(Cd,2*pi));
Xh = convert(HarmonicArray,PeriodicFunctionMatrix(X,8*pi));
Xderh = convert(HarmonicArray,PeriodicFunctionMatrix(Xder,16*pi));
@test Ah*Xh+Xh*Ah'+Ch ≈  derivative(Xh) ≈ Xderh
@test Ah'*Xh+Xh*Ah+Cdh ≈ -derivative(Xh) 
@test norm(Ah*Xh+Xh*Ah'+Ch-derivative(Xh),Inf) < 1.e-7 && norm(derivative(Xh)- Xderh) < 1.e-7
@test norm(Ah'*Xh+Xh*Ah+Cdh+derivative(Xh),1) < 1.e-7

# PeriodicSymbolicMatrix
@variables t
A11 = [0  1; -10*cos(t)-1 -24-19*sin(t)]
X1 =  [1+cos(t) 0; 0 1+sin(t)] 
X1der = [-sin(t) 0; 0 cos(t)] 
As = PeriodicSymbolicMatrix(A11,2*pi)
Cs = PeriodicSymbolicMatrix(X1der - A11*X1-X1*A11', 2*pi)
Cds = PeriodicSymbolicMatrix(-(A11'*X1+X1*A11+X1der),2*pi)
Xs = PeriodicSymbolicMatrix(X1,2*pi)
Xders = PeriodicSymbolicMatrix(X1der,2*pi)

@test issymmetric(Cs) && issymmetric(Cds) && issymmetric(Xs) && issymmetric(Xders)
@test As*Xs+Xs*As'+Cs ==  derivative(Xs) == Xders
@test As'*Xs+Xs*As+Cds == -derivative(Xs) 
@test norm(As*Xs+Xs*As'+Cs - derivative(Xs),Inf) < 1.e-7 && norm(derivative(Xs)- Xders) < 1.e-7
@test norm(As'*Xs+Xs*As+Cds + derivative(Xs),1) < 1.e-7
@test As*Xs+Xs*As'+Cs ≈  derivative(Xs) ≈ Xders
@test As'*Xs+Xs*As+Cds ≈ -derivative(Xs) 

D = rand(2,2)
@test As+I == I+As && As*5 == 5*As && As*D ≈ -As*(-D) && iszero(As-As) && !iszero(As)
@test PeriodicSymbolicMatrix(D,2*pi) == PeriodicSymbolicMatrix(D,4*pi) && 
      PeriodicSymbolicMatrix(D,2*pi) ≈ PeriodicSymbolicMatrix(D,4*pi)

As = PeriodicSymbolicMatrix(A11,4*pi,nperiod=2)
Cs = PeriodicSymbolicMatrix(X1der - A11*X1-X1*A11', 2*pi)
Cds = PeriodicSymbolicMatrix(-(A11'*X1+X1*A11+X1der),2*pi)
Xs = PeriodicSymbolicMatrix(X1,8*pi,nperiod=4)
Xders = PeriodicSymbolicMatrix(X1der,16*pi,nperiod = 8)
@test As*Xs+Xs*As'+Cs ==  derivative(Xs) == Xders
@test As'*Xs+Xs*As+Cds == -derivative(Xs)
@test As*Xs+Xs*As'+Cs ≈  derivative(Xs) ≈ Xders
@test As'*Xs+Xs*As+Cds ≈ -derivative(Xs)

@test inv(As)*As ≈ I ≈ As*inv(As) 
@test inv(As)*As == I == As*inv(As) 


# FourierFunctionMatrix
Af = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(A,2*pi));
Af1 = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(A1,2*pi));
Cf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(X,2*pi));
Xderf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Xder,2*pi));
@test issymmetric(Cf) && issymmetric(Cdf) && issymmetric(Xf) && issymmetric(Xderf)
@test Af*Xf+Xf*Af'+Cf ≈  derivative(Xf) ≈ Xderf
@test Af'*Xf+Xf*Af+Cdf ≈ -derivative(Xf) 
@test norm(Af*Xf+Xf*Af'+Cf-derivative(Xf),Inf) < 1.e-7 && norm(derivative(Xf)- Xderf) < 1.e-7
@test norm(Af'*Xf+Xf*Af+Cdf+derivative(Xf),1) < 1.e-7

D = rand(2,2)
@test Af+I == I+Af && Af*5 == 5*Af && Af*D ≈ -Af*(-D)  && iszero(Af-Af) && !iszero(Af)
@test FourierFunctionMatrix(D,2*pi) == FourierFunctionMatrix(D,4*pi) && 
      FourierFunctionMatrix(D,2*pi) ≈ FourierFunctionMatrix(D,4*pi)
@test inv(Af1)*Af1 ≈ I ≈ Af1*inv(Af1) 


Af = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(A,4*pi,nperiod=2));
Cf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(X,8*pi,nperiod=4));
Xderf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Xder,16*pi,nperiod=8));
@test Af*Xf+Xf*Af'+Cf ≈  derivative(Xf) ≈ Xderf
@test Af'*Xf+Xf*Af+Cdf ≈ -derivative(Xf) 
@test norm(Af*Xf+Xf*Af'+Cf-derivative(Xf),Inf) < 1.e-7 && norm(derivative(Xf)- Xderf) < 1.e-7
@test norm(Af'*Xf+Xf*Af+Cdf+derivative(Xf),1) < 1.e-7

# PeriodicTimeSeriesMatrix
Ats = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(A,2*pi));
Cts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(X,2*pi));
Xderts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Xder,2*pi));
@test issymmetric(Cts) && issymmetric(Cdts) && issymmetric(Xts) && issymmetric(Xderts)
@test Ats*Xts+Xts*Ats'+Cts ≈  derivative(Xts) ≈ Xderts
@test Ats'*Xts+Xts*Ats+Cdts ≈ -derivative(Xts) 
@test norm(Ats*Xts+Xts*Ats'+Cts-derivative(Xts),Inf) < 1.e-7 && norm(derivative(Xts)- Xderts) < 1.e-7
@test norm(Ats'*Xts+Xts*Ats+Cdts+derivative(Xts),1) < 1.e-7

D = rand(2,2)
@test Ats+I == I+Ats && Ats*5 == 5*Ats && Ats*D ≈ -Ats*(-D)  && iszero(Ats-Ats) && !iszero(Ats)
@test PeriodicTimeSeriesMatrix(D,2*pi) == PeriodicTimeSeriesMatrix(D,4*pi) && 
      PeriodicTimeSeriesMatrix(D,2*pi) ≈ PeriodicTimeSeriesMatrix(D,4*pi)
@test inv(Ats)*Ats ≈ I ≈ Ats*inv(Ats) 


Ats = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(A,4*pi,nperiod=2));
Cts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(X,8*pi,nperiod=4));
Xderts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Xder,16*pi,nperiod=8));
@test Ats*Xts+Xts*Ats'+Cts ≈  derivative(Xts) ≈ Xderts
@test Ats'*Xts+Xts*Ats+Cdts ≈ -derivative(Xts) 
@test norm(Ats*Xts+Xts*Ats'+Cts-derivative(Xts),Inf) < 1.e-7 && norm(derivative(Xts)- Xderts) < 1.e-7
@test norm(Ats'*Xts+Xts*Ats+Cdts+derivative(Xts),1) < 1.e-7


# PeriodicArray
n = 5; pa = 3; px = 6;   
Ad = 0.5*PeriodicArray(rand(Float64,n,n,pa),pa);
x = rand(n,n,px); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:px];
Xd = PeriodicArray(x,px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pflyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr


Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad

D = rand(n,n)
@test Ad*5 == 5*Ad && Ad*D ≈ -Ad*(-D)  && iszero(Ad-Ad) && !iszero(Ad)
@test PeriodicArray(D,2*pi) == PeriodicArray(D,4*pi) && 
      PeriodicArray(D,2*pi) ≈ PeriodicArray(D,4*pi)


Ad1 = PeriodicArray(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicArray(x,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1+Qdr1)
Xf1 = pflyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1

# PeriodicMatrix
n = 5; pa = 3; px = 6;   
Ad = 0.5*PeriodicMatrix([rand(Float64,n,n) for i in 1:pa],pa);
x = [rand(n,n) for i in 1:px]
Xd = PeriodicMatrix([ x[i]+x[i]' for i in 1:px],px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pflyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr


Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad

D = rand(n,n)
@test Ad*5 == 5*Ad  && Ad*D ≈ -Ad*(-D) && iszero(Ad-Ad) && !iszero(Ad)
@test PeriodicMatrix(D,2*pi) == PeriodicMatrix(D,4*pi) && 
      PeriodicMatrix(D,2*pi) ≈ PeriodicMatrix(D,4*pi)


Ad1 = PeriodicMatrix(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicMatrix(Xd.M,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1+Qdr1)
Xf1 = pflyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1

# time-varying dimensions
na = [5, 3, 3, 4, 1]; ma = [3, 3, 4, 1, 5]; pa = 5; px = 5;   
#na = 5*na; ma = 5*ma;
Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
x = [rand(na[i],na[i]) for i in 1:px]
Xd = PeriodicMatrix([ x[i]+x[i]' for i in 1:px],px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pflyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr

Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')

@test Ad*5 == 5*Ad &&  iszero(Ad-Ad) && !iszero(Ad) && Qdf + I == I+Qdf
@test_throws DimensionMismatch Ad ≈ I && Ad-I 


Ad1 = PeriodicMatrix(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicMatrix(Xd.M,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1)&& issymmetric(Qdr1)
Xf1 = pflyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1


end # pmops

end