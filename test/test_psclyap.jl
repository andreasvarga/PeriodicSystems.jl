module Test_psclyap

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using ApproxFun
using Symbolics

println("Test_pclyap")

@testset "pclyap" begin


# generate symbolic periodic matrices
@variables t
A1 = [0  1; -10*cos(t)-1 -24-19*sin(t)]
As = PeriodicSymbolicMatrix(A1,2*pi)
Xs =  [1+cos(t) 0; 0 1+sin(t)] 
Xdots = [-sin(t) 0; 0 cos(t)] 
Cds = PeriodicSymbolicMatrix(-(A1'*Xs+Xs*A1+Xdots),2*pi)
Cs = PeriodicSymbolicMatrix(Xdots - A1*Xs-Xs*A1', 2*pi)

@time Ys = pclyap(As,Cs; K = 100, reltol = 1.e-10);
@test substitute.(norm(Xs-Ys.F), (Dict(t => rand()),))[1] < 1.e-7

Ls = As.F*Ys.F+Ys.F*transpose(As.F)+Cs.F.-Symbolics.derivative(Ys.F,t)
@test norm(substitute.(Ls, (Dict(t => rand()),)),Inf) < 1.e-7

@time Ys = pclyap(As,Cds; adj = true, K = 100, reltol = 1.e-10);
@test substitute.(norm(Xs-Ys.F), (Dict(t => rand()),))[1] < 1.e-7

Lds = transpose(As.F)*Ys.F+Ys.F*As.F+Cds.F.+Symbolics.derivative(Ys.F,t)
@test norm(substitute.(Lds, (Dict(t => rand()),)),Inf) < 1.e-7

@time Ys = pfclyap(As,Cs; K = 100, reltol = 1.e-10);
@test substitute.(norm(Xs-Ys.F), (Dict(t => rand()),))[1] < 1.e-7

@time Ys = prclyap(As,Cds; K = 100, reltol = 1.e-10);
@test substitute.(norm(Xs-Ys.F), (Dict(t => rand()),))[1] < 1.e-7


# generate periodic function matrices
A(t) = [0  1; -10*cos(t)-1 -24-19*sin(t)]
X(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
Xdot(t) = [-sin(t) 0; 0 cos(t)]  # derivative of the desired solution
C(t) = [ -sin(t)  -1-sin(t)-(-1-10cos(t))*(1+cos(t));
-1-sin(t)-(-1-10cos(t))*(1+cos(t))   cos(t)- 2(-24 - 19sin(t))*(1 + sin(t)) ]  # corresponding C
Cd(t) = [ sin(t)  -1-cos(t)-(-1-10cos(t))*(1+sin(t));
-1-cos(t)-(-1-10cos(t))*(1+sin(t))    -cos(t)-2(-24-19sin(t))*(1 + sin(t)) ] # corresponding Cd
At = PeriodicFunctionMatrix(A,2*pi)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,2*pi)

tt = Vector((1:500)*2*pi/500) 
@time Yt = pclyap(At, Ct, K = 500, reltol = 1.e-10);
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

del = Yt.period*sqrt(eps())
Lt = t -> At.f(t)*Yt.f(t)+Yt.f(t)*At.f(t)'+Ct.f(t)-(Yt.f(t+del)-Yt.f(t-del))/(2*del)
@test maximum(norm.(Lt.(tt),Inf)) .< 1.e-7

@time Yt = pclyap(At, Cdt, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

Ldt = t -> At.f(t)'*Yt.f(t)+Yt.f(t)*At.f(t)+Cdt.f(t)+(Yt.f(t+del)-Yt.f(t-del))/(2*del)
@test maximum(norm.(Ldt.(tt),Inf)) .< 1.e-7

@time Y = pfclyap(At, Ct, K = 500, reltol = 1.e-10);
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = prclyap(At, Cdt, K = 500, reltol = 1.e-10)
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

# solve using harmonic arrays
tt = Vector((1:500)*2*pi/500) 
@time Y = pclyap(convert(HarmonicArray,At), convert(HarmonicArray,Ct), K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

@time Y = pclyap(convert(HarmonicArray,At), convert(HarmonicArray,Cdt), K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

@time Y = pfclyap(convert(HarmonicArray,At), convert(HarmonicArray,Ct), K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

@time Y = prclyap(convert(HarmonicArray,At), convert(HarmonicArray,Cdt), K = 500, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7


# solve using Fourier function matrices
Af = convert(FourierFunctionMatrix,At);
Cf = convert(FourierFunctionMatrix,Ct); 
Cdf = convert(FourierFunctionMatrix,Cdt)
tt = Vector((1:500)*2*pi/500) 
@time Yf = pclyap(Af, Cf, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yf,tt).-Xt.f.(tt))) < 1.e-7

Lf = Af.M*Yf.M+Yf.M*transpose(Af.M)+Cf.M-differentiate(Yf.M);     
@test norm(Lf) < 1.e-7

@time Y = pclyap(Af, Cdf, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

Ldf = transpose(Af.M)*Yf.M+Yf.M*Af.M+Cdf.M+differentiate(Yf.M);     
@test norm(Ldf) < 1.e-7


@time Y = pfclyap(convert(FourierFunctionMatrix,At), convert(FourierFunctionMatrix,Ct), K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

@time Y = prclyap(convert(FourierFunctionMatrix,At), convert(FourierFunctionMatrix,Cdt), K = 500, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7


# solve using periodic time-series matrices
Ats = convert(PeriodicTimeSeriesMatrix,At);
Cts = convert(PeriodicTimeSeriesMatrix,Ct);
Cdts = convert(PeriodicTimeSeriesMatrix,Cdt);
tt = Vector((1:500)*2*pi/500) 
@time Yts = pclyap(Ats, Cts, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7

@time Yts = pclyap(Ats, Cdts, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7

@time Yts = pfclyap(Ats, Cts, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7

@time Yts = prclyap(Ats, Cdts, K = 500, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7



for K in (1, 16, 64, 128, 256)
    @time Y = pclyap(At,Ct; K, reltol = 1.e-10);    
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for K in (1, 16, 64, 128, 256)
    @time Y = pclyap(At,Cdt; K, adj = true, reltol = 1.e-10);   
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for (na,nc) in ((1,1),(1,2),(2,1),(2,2))
   At = PeriodicFunctionMatrix(A,4*pi; nperiod = na)
   Ct = PeriodicFunctionMatrix(C,4*pi; nperiod = nc)
   Cdt = PeriodicFunctionMatrix(Cd,4*pi; nperiod = nc)
   Xt = PeriodicFunctionMatrix(X,4*pi; nperiod = gcd(na,nc))

   @time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10);
   
   tt = Vector((1:500)*4*pi/500/Y.nperiod) 
   @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

   @time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10);
   
   tt = Vector((1:500)*4*pi/500/Y.nperiod) 
   @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

end

# # generate symbolic periodic matrices
# using Symbolics
# @variables t
# As = [0  1; -1 -24]
# Xs =  [1+cos(t) 0; 0 1+sin(t)] 
# Xdots = [-sin(t) 0; 0 cos(t)] 
# Cds = -(As'*Xs+Xs*As+Xdots)
# Cs = Xdots - As*Xs-Xs*As'

A1 = [0  1; -1 -24]
X1(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
X1dot(t) = [-sin(t) 0; 0 cos(t)]  # derivative of the desired solution
C1(t) = [ -sin(t)   cos(t)-sin(t);
cos(t)-sin(t)  48+48sin(t)+cos(t) ]  # corresponding C
Cd1(t) = [ sin(t)   -cos(t)+sin(t);
-cos(t)+sin(t)  48+48sin(t)-cos(t)  ] # corresponding Cd


At = PeriodicFunctionMatrix(A1,2*pi)
Ct = PeriodicFunctionMatrix(C1,2*pi)
Cdt = PeriodicFunctionMatrix(Cd1,2*pi)
Xt = PeriodicFunctionMatrix(X1,2*pi)

@time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7


A2 = [0  1; -1 -24]
X2 = [1 0; 0 1]  # desired solution
X2dot = [0 0; 0 0]  # derivative of the desired solution
C2 = [ 0   0; 0  48 ]  # corresponding C
Cd2 = [ 0   0; 0  48 ] # corresponding Cd

At = PeriodicFunctionMatrix(A2,2*pi)
Ct = PeriodicFunctionMatrix(C2,2*pi)
Cdt = PeriodicFunctionMatrix(Cd2,2*pi)
Xt = PeriodicFunctionMatrix(X2,2*pi)

@time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

end # psclyap

end
