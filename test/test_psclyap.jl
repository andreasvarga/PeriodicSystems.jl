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
@test norm(At*Yt+Yt*At'+Ct-derivative(Yt)) < 1.e-7


@time Yt = pclyap(At, Cdt, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7
@test norm(At'*Yt+Yt*At+Cdt+derivative(Yt)) < 1.e-7

@time Y = pfclyap(At, Ct, K = 500, reltol = 1.e-10);
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = prclyap(At, Cdt, K = 500, reltol = 1.e-10)
@test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-7

solver = "non-stiff"
for solver in ("non-stiff", "stiff", "symplectic", "noidea")
    println("solver = $solver")
    @time Yt = pclyap(At, Ct; solver, K = 500, reltol = 1.e-10, abstol = 1.e-10);
    @test maximum(norm.(Yt.f.(tt).-Xt.f.(tt))) < 1.e-6
end


# solve using harmonic arrays
tt = Vector((1:500)*2*pi/500) 
Ah = convert(HarmonicArray,At);
Ch = convert(HarmonicArray,Ct);
Cdh = convert(HarmonicArray,Cdt);
Xh = convert(HarmonicArray,Xt);

@time Yh = pclyap(Ah, Ch, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yh,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Ah*Yh+Yh*Ah'+Ch-derivative(Yh)) < 1.e-7

@time Y = pclyap(Ah, Cdh, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Ah'*Yh+Yh*Ah+Cdh+derivative(Yh)) < 1.e-7

@time Y = pfclyap(Ah, Ch, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7

@time Y = prclyap(Ah, Cdh, K = 500, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Y,tt).-Xt.f.(tt))) < 1.e-7


# solve using Fourier function matrices
Af = convert(FourierFunctionMatrix,At);
Cf = convert(FourierFunctionMatrix,Ct); 
Cdf = convert(FourierFunctionMatrix,Cdt)
Xf = convert(FourierFunctionMatrix,Xt);
tt = Vector((1:500)*2*pi/500) 
@time Yf = pclyap(Af, Cf, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yf,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Af*Yf+Yf*Af'+Cf-derivative(Yf)) < 1.e-7

@time Yf = pclyap(Af, Cdf, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Yf,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Af'*Yf+Yf*Af+Cdf+derivative(Yf)) < 1.e-7

@time Yf = pfclyap(convert(FourierFunctionMatrix,At), convert(FourierFunctionMatrix,Ct), K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yf,tt).-Xt.f.(tt))) < 1.e-7

@time Yf = prclyap(convert(FourierFunctionMatrix,At), convert(FourierFunctionMatrix,Cdt), K = 500, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Yf,tt).-Xt.f.(tt))) < 1.e-7


# solve using periodic time-series matrices
Ats = convert(PeriodicTimeSeriesMatrix,At);
Cts = convert(PeriodicTimeSeriesMatrix,Ct);
Cdts = convert(PeriodicTimeSeriesMatrix,Cdt);
tt = Vector((1:500)*2*pi/500) 
@time Yts = pclyap(Ats, Cts, K = 500, reltol = 1.e-10);
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Ats*Yts+Yts*Ats'+Cts-derivative(Yts)) < 1.e-7

@time Yts = pclyap(Ats, Cdts, K = 500, adj = true, reltol = 1.e-10)
@test maximum(norm.(tvmeval(Yts,tt).-Xt.f.(tt))) < 1.e-7
@test norm(Ats'*Yts+Yts*Ats+Cdts+derivative(Yts)) < 1.e-7


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


# Pitelkau's example - singular Lyapunov equations
# ω = 0.00103448
# period = 2*pi/ω
# a = [  0            0     5.318064566757217e-02                         0
#        0            0                         0      5.318064566757217e-02
#        -1.352134256362805e-03   0             0    -7.099273035392090e-02
#        0    -7.557182037479544e-04     3.781555288420663e-02             0
# ];
# b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
# c = [1 0 0 0;0 1 0 0];
# d = zeros(2,1);
# psysc = ps(a,PeriodicFunctionMatrix(b,period),c,d);

# @time Qc = prclyap(psysc.A,psysc.B*transpose(psysc.B),K = 120, reltol = 1.e-10);
# @time Rc = pfclyap(psysc.A,transpose(psysc.C)*psysc.C, K = 120, reltol = 1.e-10);
# @test norm(Qc) > 1.e3 && norm(Rc) > 1.e3

# K = 120;
# @time psys = psc2d(psysc,period/K,reltol = 1.e-10);

# @time Qd = prlyap(psys.A,psys.B*transpose(psys.B));
# @time Rd = prlyap(transpose(psys.A),transpose(psys.C)*psys.C);
# @test norm(Qd) > 1.e3 && norm(Rd) > 1.e3

end # psclyap

end
