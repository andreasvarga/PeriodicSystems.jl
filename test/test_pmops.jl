module Test_pmops

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using ApproxFun

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

t = rand(); 
@test [At Ct](t) ≈ [At(t) Ct(t)]
@test [At; Ct](t) ≈ [At(t); Ct(t)]
@test blockdiag(At,Ct)(t) ≈ DescriptorSystems.blockdiag(At(t),Ct(t))


D = rand(2,2)
@test At+I == I+At && At*5 == 5*At && At*D ≈ -At*(-D) && iszero(At-At) && !iszero(At)
@test inv(At)*At ≈ I ≈ At*inv(At) && At+I == I+At
@test norm(At-At,1) == norm(At-At,2) == norm(At-At,Inf) == 0
@test iszero(opnorm(At-At,1)) && iszero(opnorm(At-At,2)) && iszero(opnorm(At-At,Inf)) && iszero(opnorm(At-At))
@test trace(At-At) == 0 && iszero(tr(At-At))

@test PeriodicFunctionMatrix(D,2*pi) == PeriodicFunctionMatrix(D,4*pi) && 
      PeriodicFunctionMatrix(D,2*pi) ≈ PeriodicFunctionMatrix(D,4*pi)

@test tpmeval(At,1)[1:2,1:1] == tpmeval(At[1:2,1],1) && lastindex(At,1) == 2 && lastindex(At,2) == 2

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
@test norm(Ah-Ah,1) == norm(Ah-Ah,2) == norm(Ah-Ah,Inf) == 0
@test iszero(opnorm(Ah-Ah,1)) && iszero(opnorm(Ah-Ah,2)) && iszero(opnorm(Ah-Ah,Inf)) && iszero(opnorm(Ah-Ah))
@test trace(Ah-Ah) == 0 && iszero(tr(Ah-Ah))

@test Ah1i*Ah1 ≈ I ≈ Ah1*Ah1i 
@test hrchop(Ah1i; tol = 1.e-10) ≈ hrchop(Ah1i; tol = eps()) ≈ hrchop(Ah1i; tol = 1.e-10) 
@test hrtrunc(Ah1i,19) ≈ hrtrunc(Ah1i,20)

@test blockdiag(Ah,Ch)(t) ≈ DescriptorSystems.blockdiag(Ah(t),Ch(t))

Ah = convert(HarmonicArray,PeriodicFunctionMatrix(A,4*pi));
Ch = convert(HarmonicArray,PeriodicFunctionMatrix(C,2*pi));
Cdh = convert(HarmonicArray,PeriodicFunctionMatrix(Cd,2*pi));
Xh = convert(HarmonicArray,PeriodicFunctionMatrix(X,8*pi));
Xderh = convert(HarmonicArray,PeriodicFunctionMatrix(Xder,16*pi));
@test Ah*Xh+Xh*Ah'+Ch ≈  derivative(Xh) ≈ Xderh
@test Ah'*Xh+Xh*Ah+Cdh ≈ -derivative(Xh) 
@test norm(Ah*Xh+Xh*Ah'+Ch-derivative(Xh),Inf) < 1.e-7 && norm(derivative(Xh)- Xderh) < 1.e-7
@test norm(Ah'*Xh+Xh*Ah+Cdh+derivative(Xh),1) < 1.e-7

@test tpmeval(Ah,1)[1:2,1:1] == tpmeval(Ah[1:2,1],1) && lastindex(Ah,1) == 2 && lastindex(Ah,2) == 2

t = rand(); 
@test [Ah Ch](t) ≈ [Ah(t) Ch(t)]
@test [Ah; Ch](t) ≈ [Ah(t); Ch(t)]
@test blockdiag(Ah,Ch)(t) ≈ DescriptorSystems.blockdiag(Ah(t),Ch(t))


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

@test (Symbolics.simplify(inv(As)*As) ≈ I) && (I ≈ Symbolics.simplify(As*inv(As))) 
@test inv(As)*As == I == As*inv(As) 

@test iszero(As[1:2,1:1].F - As.F[1:2,1:1]) && lastindex(As,1) == 2 && lastindex(As,2) == 2

t = rand(); 
@test [As Cs](t) ≈ [As(t) Cs(t)]
@test [As; Cs](t) ≈ [As(t); Cs(t)]
@test blockdiag(As,Cs)(t) ≈ DescriptorSystems.blockdiag(As(t),Cs(t))





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
@test norm(Af-Af,1) == norm(Af-Af,2) == norm(Af-Af,Inf) == 0
@test iszero(opnorm(Af-Af,1)) && iszero(opnorm(Af-Af,2)) && iszero(opnorm(Af-Af,Inf)) && 
      iszero(opnorm(Af-Af)) 
@test trace(Af-Af) == 0 && iszero(tr(Af-Af))

@test blockdiag(Af,Cf)(t) ≈ DescriptorSystems.blockdiag(Af(t),Cf(t))


Af = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(A,4*pi,nperiod=2));
Cf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(X,8*pi,nperiod=4));
Xderf = convert(FourierFunctionMatrix,PeriodicFunctionMatrix(Xder,16*pi,nperiod=8));
@test Af*Xf+Xf*Af'+Cf ≈  derivative(Xf) ≈ Xderf
@test Af'*Xf+Xf*Af+Cdf ≈ -derivative(Xf) 
@test norm(Af*Xf+Xf*Af'+Cf-derivative(Xf),Inf) < 1.e-7 && norm(derivative(Xf)- Xderf) < 1.e-7
@test norm(Af'*Xf+Xf*Af+Cdf+derivative(Xf),1) < 1.e-7

@test tpmeval(Af,1)[1:2,1:1] == tpmeval(Af[1:2,1],1) && lastindex(Af,1) == 2 && lastindex(Af,2) == 2

t = rand(); 
@test [Af Cf](t) ≈ [Af(t) Cf(t)]
@test [Af; Cf](t) ≈ [Af(t); Cf(t)]
@test blockdiag(Af,Cf)(t) ≈ DescriptorSystems.blockdiag(Af(t),Cf(t))

# PeriodicTimeSeriesMatrix
Ats = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(A,2*pi));
Cts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(X,2*pi));
Xderts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Xder,2*pi));
@test issymmetric(Cts) && issymmetric(Cdts) && issymmetric(Xts) && issymmetric(Xderts)
# @test Ats*Xts+Xts*Ats'+Cts ≈  derivative(Xts) ≈ Xderts
# @test Ats'*Xts+Xts*Ats+Cdts ≈ -derivative(Xts) 
@test norm(Ats*Xts+Xts*Ats'+Cts-derivative(Xts),Inf) < 1.e-6 && norm(Ats*Xts+Xts*Ats'+Cts-Xderts,Inf) < 1.e-7
@test norm(Ats'*Xts+Xts*Ats+Cdts+derivative(Xts),Inf) < 1.e-6 && norm(Ats'*Xts+Xts*Ats+Cdts+Xderts,Inf) < 1.e-6

D = rand(2,2)
@test Ats+I == I+Ats && Ats*5 == 5*Ats && Ats*D ≈ -Ats*(-D)  && iszero(Ats-Ats) && !iszero(Ats)
@test PeriodicTimeSeriesMatrix(D,2*pi) == PeriodicTimeSeriesMatrix(D,4*pi) && 
      PeriodicTimeSeriesMatrix(D,2*pi) ≈ PeriodicTimeSeriesMatrix(D,4*pi)
@test inv(Ats)*Ats ≈ I ≈ Ats*inv(Ats) 
@test norm(Ats-Ats,1) == norm(Ats-Ats,2) == norm(Ats-Ats,Inf) == 0
@test iszero(opnorm(Ats-Ats,1)) && iszero(opnorm(Ats-Ats,2)) && iszero(opnorm(Ats-Ats,Inf)) && 
      iszero(opnorm(Ats-Ats)) 
@test trace(Ats-Ats) == 0 && iszero(tr(Ats-Ats))

t = rand(); 
@test blockdiag(Ats,Cts)(t) ≈ DescriptorSystems.blockdiag(Ats(t),Cts(t))

# same tsub
# TA = PeriodicTimeSeriesMatrix([[i;;] for i in 1:4],30,nperiod=6)
# TB = PeriodicTimeSeriesMatrix([[i;;] for i in 1:2],10,nperiod=2)
TA = PeriodicTimeSeriesMatrix([rand(2,2) for i in 1:4],30,nperiod=6)
TB = PeriodicTimeSeriesMatrix([rand(2,2) for i in 1:2],10,nperiod=2)
AB = TA+TB
ts = sort(rand(100)*AB.period); 
@test norm(AB.(ts).-TA.(ts).-TB.(ts)) < 1.e-10
AB = TA*TB
ts = sort(rand(10)*AB.period); 
@test norm(AB.(ts).-TA.(ts).*TB.(ts)) < 1.e-10


# same period
# TA = PeriodicTimeSeriesMatrix([[i;;] for i in 1:4],30,nperiod=6)
# TB = PeriodicTimeSeriesMatrix([[i;;] for i in 1:2],30,nperiod=2)
TA = PeriodicTimeSeriesMatrix([rand(2,2) for i in 1:4],30,nperiod=6)
TB = PeriodicTimeSeriesMatrix([rand(2,2) for i in 1:2],30,nperiod=2)
AB = TA+TB
ts = sort(rand(10)*AB.period); 
@test norm(AB.(ts).-TA.(ts).-TB.(ts)) < 1.e-10
AB = TA*TB
@test norm(AB.(ts).-TA.(ts).*TB.(ts)) < 1.e-10



Ats = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(A,4*pi,nperiod=2));
Cts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(C,2*pi));
Cdts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Cd,2*pi));
Xts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(X,8*pi,nperiod=4));
Xderts = convert(PeriodicTimeSeriesMatrix,PeriodicFunctionMatrix(Xder,16*pi,nperiod=8));
@test norm(Ats*Xts+Xts*Ats'+Cts-derivative(Xts),Inf) < 1.e-6 && norm(Ats*Xts+Xts*Ats'+Cts-Xderts,Inf) < 1.e-6 
@test norm(Ats'*Xts+Xts*Ats+Cdts+derivative(Xts),Inf) < 1.e-6 && norm(Ats'*Xts+Xts*Ats+Cdts+Xderts,Inf) < 1.e-6

@test Ats[1:2,1:1].values == [Ats.values[i][1:2,1:1] for i in 1:length(Ats)] && lastindex(Ats,1) == 2 && lastindex(Ats,2) == 2

t = rand(); 
@test [Ats Cts](t) ≈ [Ats(t) Cts(t)]
@test [Ats; Cts](t) ≈ [Ats(t); Cts(t)]
@test blockdiag(Ats,Cts)(t) ≈ DescriptorSystems.blockdiag(Ats(t),Cts(t))


# PeriodicSwitchingMatrix
t1 = -rand(2,2); t2 = rand(2,2); Asw = PeriodicSwitchingMatrix([t1,t2],[0.,1.],2)
t1 = rand(2,2); t2 = rand(2,2); Csw = PeriodicSwitchingMatrix([t1,t2],[0.,1.5],2)
@test Csw == convert(PeriodicSwitchingMatrix,convert(PeriodicTimeSeriesMatrix,Csw,ns=10))
t1 = rand(2,2); t2 = rand(2,2); Tsw = PeriodicSwitchingMatrix([t1,t2],[0.,pi/2],2)
@test norm((Tsw - convert(PeriodicSwitchingMatrix,convert(PeriodicTimeSeriesMatrix,Tsw,ns=10))).(rand(10*2))) == 0

t = 2*rand(); 
@test (Asw+Csw)(t) ≈ Asw(t)+Csw(t)
@test (Asw*Csw)(t) ≈ Asw(t)*Csw(t)
@test [Asw Csw](t) ≈ [Asw(t) Csw(t)]
@test [Asw; Csw](t) ≈ [Asw(t); Csw(t)]
@test norm(Asw-2*Asw+Asw) == 0
D = rand(2,2)
@test Asw+I == I+Asw && Asw*5 == 5*Asw && Asw*D ≈ -Asw*(-D)  && iszero(Asw-Asw) && !iszero(Asw)
@test PeriodicSwitchingMatrix(D,2) == PeriodicSwitchingMatrix(D,4) && 
      PeriodicSwitchingMatrix(D,2) ≈ PeriodicSwitchingMatrix(D,4)
@test issymmetric(Csw*Csw')
@test inv(Asw)*Asw ≈ I ≈ Asw*inv(Asw) 
@test norm(Asw-Asw,1) == norm(Asw-Asw,2) == norm(Asw-Asw,Inf) == 0
@test iszero(opnorm(Asw-Asw,1)) && iszero(opnorm(Asw-Asw,2)) && iszero(opnorm(Asw-Asw,Inf)) && 
      iszero(opnorm(Asw-Asw)) 
@test trace(Asw-Asw) == 0 && iszero(tr(Asw-Asw))
t = rand(); 
@test blockdiag(Asw,Csw)(t) ≈ DescriptorSystems.blockdiag(Asw(t),Csw(t))




# PeriodicArray
n = 5; pa = 3; px = 6;   
Ad = 0.5*PeriodicArray(rand(Float64,n,n,pa),pa);
x = rand(n,n,px); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:px];
Xd = PeriodicArray(x,px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pfdlyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prdlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr


Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad
@test norm(Ad-Ad,1) == norm(Ad-Ad,2) == norm(Ad-Ad,Inf) == 0
@test iszero(opnorm(Ad-Ad,1)) && iszero(opnorm(Ad-Ad,2)) && iszero(opnorm(Ad-Ad,Inf))
@test trace(Ad-Ad) == 0 && iszero(tr(Ad-Ad))


D = rand(n,n)
@test Ad*5 == 5*Ad && Ad*D ≈ -Ad*(-D)  && iszero(Ad-Ad) && !iszero(Ad)
@test PeriodicArray(D,2*pi) == PeriodicArray(D,4*pi) && 
      PeriodicArray(D,2*pi) ≈ PeriodicArray(D,4*pi)

@test blockdiag(Ad,Xd)[10] ≈ DescriptorSystems.blockdiag(Ad[10],Xd[10])      


Ad1 = PeriodicArray(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicArray(x,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1+Qdr1)
Xf1 = pfdlyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prdlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1

@test Ad[1:2,1:1].M == Ad.M[1:2,1:1,:]  && lastindex(Ad,1) == n && lastindex(Ad,2) == n

@test blockdiag(Ad1,Xd1)[10] ≈ DescriptorSystems.blockdiag(Ad1[10],Xd1[10])  


# PeriodicMatrix
n = 5; pa = 3; px = 6;   
Ad = 0.5*PeriodicMatrix([rand(Float64,n,n) for i in 1:pa],pa);
x = [rand(n,n) for i in 1:px]
Xd = PeriodicMatrix([ x[i]+x[i]' for i in 1:px],px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pfdlyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prdlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr


Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad
@test norm(Ad-Ad,1) == norm(Ad-Ad,2) == norm(Ad-Ad,Inf) == 0
@test iszero(opnorm(Ad-Ad,1)) && iszero(opnorm(Ad-Ad,2)) && iszero(opnorm(Ad-Ad,Inf))
@test trace(Ad-Ad) == 0 && iszero(tr(Ad-Ad))


D = rand(n,n)
@test Ad*5 == 5*Ad  && Ad*D ≈ -Ad*(-D) && iszero(Ad-Ad) && !iszero(Ad)
@test PeriodicMatrix(D,2*pi) == PeriodicMatrix(D,4*pi) && 
      PeriodicMatrix(D,2*pi) ≈ PeriodicMatrix(D,4*pi)
     
@test blockdiag(Ad,Xd)[10] ≈ DescriptorSystems.blockdiag(Ad[10],Xd[10])      
      


Ad1 = PeriodicMatrix(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicMatrix(Xd.M,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1+Qdr1)
Xf1 = pfdlyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prdlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1

@test Ad[1:2,1:1].M == [Ad.M[i][1:2,1:1] for i in 1:length(Ad)] && lastindex(Ad,1) == n && lastindex(Ad,2) == n


@test [[Ad Ad]; [Ad Ad]] == [[Ad;Ad] [Ad;Ad]]
@test blockdiag(Ad1,Xd1)[10] ≈ DescriptorSystems.blockdiag(Ad1[10],Xd1[10]) 


# time-varying dimensions
na = [5, 3, 3, 4, 1]; ma = [3, 3, 4, 1, 5]; pa = 5; px = 5;   
#na = 5*na; ma = 5*ma;
Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
x = [rand(na[i],na[i]) for i in 1:px]
Xd = PeriodicMatrix([ x[i]+x[i]' for i in 1:px],px);
Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pfdlyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prdlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr

Qds = pmshift(Qdf); 

@test issymmetric(Qdf) && issymmetric(Qds) && isequal(pmshift(pmshift(Qdf,1),-1),Qdf) && iszero(Qdf-Qdf')

@test Ad*5 == 5*Ad &&  iszero(Ad-Ad) && !iszero(Ad) && Qdf + I == I+Qdf
@test_throws DimensionMismatch Ad ≈ I && Ad-I 

@test blockdiag(Ad,Xd)[10] ≈ DescriptorSystems.blockdiag(Ad[10],Xd[10])      



Ad1 = PeriodicMatrix(Ad.M,2*pa;nperiod=2);
Xd1 = PeriodicMatrix(Xd.M,3*px; nperiod = 3);
Qdf1 = -Ad1*Xd1*Ad1'+pmshift(Xd1); Qdf1 = (Qdf1+transpose(Qdf1))/2
Qdr1 = -Ad1'*pmshift(Xd1)*Ad1+Xd1; Qdr1 = (Qdr1+transpose(Qdr1))/2

@test Ad1 == Ad && (Ad1+Ad)/3 ≈ 2*Ad/3 && issymmetric(Qdf1)&& issymmetric(Qdr1)
Xf1 = pfdlyap(Ad1, Qdf1);
@test Ad*Xf1*Ad' + Qdf ≈ pmshift(Xf1) && Xd ≈ Xf1
Xr1 = prdlyap(Ad1, Qdr1);
@test Ad'*pmshift(Xr1)*Ad + Qdr ≈ Xr1 && Xd ≈ Xr1

@test Ad[1:1,1:1].M == [Ad.M[i][1:1,1:1] for i in 1:length(Ad)] && lastindex(Ad,1) == 1 && lastindex(Ad,2) == 1

@test blockdiag(Ad1,Xd1)[10] ≈ DescriptorSystems.blockdiag(Ad1[10],Xd1[10]) 


# SwitchingPeriodicMatrix
n = 2; pa = 3; px = 6; T = 10; 
Ad = 0.5*SwitchingPeriodicMatrix([rand(Float64,n,n) for i in 1:pa],[10,15,20],T);
x = [rand(n,n) for i in 1:px]
Xd = SwitchingPeriodicMatrix([ x[i]+x[i]' for i in 1:px],[2, 3, 5,7, 9, 10],T;nperiod=2);
@test Ad.Ts == Xd.Ts

@test Ad == pmshift(pmshift(Ad),-1)
@test Ad == pmshift(pmshift(Ad,10),-10)

Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pfdlyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prdlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr



@test issymmetric(Xd) && iszero(Xd-Xd')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad
@test Ad == reverse(reverse(Ad))
@test Ad == convert(SwitchingPeriodicMatrix,convert(PeriodicMatrix,Ad))
@test Ad ≈ convert(SwitchingPeriodicMatrix,convert(PeriodicMatrix,Ad))
@test norm(Ad,Inf) == norm(convert(PeriodicMatrix,Ad),Inf)
@test norm(Ad,1) ≈ norm(convert(PeriodicMatrix,Ad),1)
@test norm(Ad,2) ≈ norm(convert(PeriodicMatrix,Ad),2)
@test iszero(opnorm(Ad-Ad,Inf)) && iszero(opnorm(Ad-Ad,1)) && iszero(opnorm(Ad-Ad,2))
@test trace(Ad) ≈ trace(convert(PeriodicMatrix,Ad)) && tr(Ad) ≈ convert(SwitchingPeriodicMatrix,tr(convert(PeriodicMatrix,Ad)))



D = rand(n,n)
@test Ad*5 == 5*Ad  && Ad*D ≈ -Ad*(-D) && iszero(Ad-Ad) && !iszero(Ad)
@test SwitchingPeriodicMatrix(D,2*pi) == SwitchingPeriodicMatrix(D,4*pi) && 
      SwitchingPeriodicMatrix(D,2*pi) ≈ SwitchingPeriodicMatrix(D,4*pi)


@test Ad[1:2,1:1].M == [Ad.M[i][1:2,1:1] for i in 1:length(Ad.M)] && lastindex(Ad,1) == n && lastindex(Ad,2) == n

@test [[Ad Xd]; [Xd Ad]] == [[Ad;Xd] [Xd;Ad]]

@test blockdiag(Ad,Xd)[10] ≈ DescriptorSystems.blockdiag(Ad[10],Xd[10])   

# SwitchingPeriodicArray
n = 2; pa = 3; px = 6; T = 10; 
Ad = 0.5*SwitchingPeriodicArray(rand(Float64,n,n,pa),[10,15,20],T);
x = pmsymadd!(PeriodicArray(rand(n,n,px),T));
Xd = SwitchingPeriodicArray(x.M,[2, 3, 5,7, 9, 10],T;nperiod=2);
@test Ad.Ts == Xd.Ts

@test Ad == pmshift(pmshift(Ad),-1)
@test Ad == pmshift(pmshift(Ad,10),-10)

Qdf = -Ad*Xd*Ad'+pmshift(Xd); Qdf = (Qdf+transpose(Qdf))/2
Qdr = -Ad'*pmshift(Xd)*Ad+Xd; Qdr = (Qdr+transpose(Qdr))/2

Xf = pfdlyap(Ad, Qdf);
@test Ad*Xf*Ad' + Qdf ≈ pmshift(Xf) && Xd ≈ Xf
Xr = prdlyap(Ad, Qdr);
@test Ad'*pmshift(Xr)*Ad + Qdr ≈ Xr && Xd ≈ Xr



@test issymmetric(Xd) && iszero(Xd-Xd')
@test inv(Ad)*Ad ≈ I ≈ Ad*inv(Ad) && Ad+I == I+Ad
@test Ad == reverse(reverse(Ad))
@test Ad == convert(SwitchingPeriodicArray,convert(PeriodicArray,Ad))
@test Ad ≈ convert(SwitchingPeriodicArray,convert(PeriodicArray,Ad))
@test norm(Ad,Inf) == norm(convert(PeriodicArray,Ad),Inf)
@test norm(Ad,1) ≈ norm(convert(PeriodicArray,Ad),1)
@test norm(Ad,2) ≈ norm(convert(PeriodicArray,Ad),2)
@test iszero(opnorm(Ad-Ad,Inf)) && iszero(opnorm(Ad-Ad,1)) && iszero(opnorm(Ad-Ad,2))
@test trace(Ad) ≈ trace(convert(PeriodicArray,Ad)) && tr(Ad) ≈ convert(SwitchingPeriodicArray,tr(convert(PeriodicArray,Ad)))


D = rand(n,n)
@test Ad*5 == 5*Ad  && Ad*D ≈ -Ad*(-D) && iszero(Ad-Ad) && !iszero(Ad)
@test SwitchingPeriodicArray(D,2*pi) == SwitchingPeriodicArray(D,4*pi) && 
      SwitchingPeriodicArray(D,2*pi) ≈ SwitchingPeriodicArray(D,4*pi)


@test Ad[1:2,1:1].M == Ad.M[1:2,1:1,:] && lastindex(Ad,1) == n && lastindex(Ad,2) == n

@test [[Ad Xd]; [Xd Ad]] == [[Ad;Xd] [Xd;Ad]]

@test blockdiag(Ad,Xd)[10] ≈ DescriptorSystems.blockdiag(Ad[10],Xd[10])   


end # pmops

end