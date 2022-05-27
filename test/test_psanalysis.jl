module Test_psanalysis

using PeriodicSystems
using DescriptorSystems
using Symbolics
using Test
using LinearAlgebra
using ApproxFun

println("Test_analysis")

# # selected sequence to illustrate failure of BlockMatrices.jl v0.16.16
# using LinearAlgebra
# using ApproxFun
# Af = Fun(t -> [0 1; -10*cos(t) -24-10*sin(t)],Fourier(0..2π));
# D = Derivative(domain(Af));
# ND = [D 0I; 0I D];
# Aop = Af - ND;
# NA = 322
# RW = Aop[1:NA,1:NA]

@testset "test_analysis" begin

# using Floquet based approach
At = PeriodicFunctionMatrix(t -> [0 1; -10*cos(t) -24-10*sin(t)],2pi);
psys = ps(At,HarmonicArray(rand(2,1),2*pi),rand(1,2))
@time ev = pspole(psys,500)
@test sort(ev) ≈ sort([0;-24])

# using Fourier series
Afun = FourierFunctionMatrix(Fun(t -> [0 1; -10*cos(t) -24-10*sin(t)],Fourier(0..2π)))
psys = ps(Afun,FourierFunctionMatrix(rand(2,1),2*pi),rand(1,2))
@time ev1 = pspole(psys,50)
@test isapprox(sort(real(ev)),sort(real(ev1)),rtol=1.e-6) && norm(imag(ev1)) < 1.e-10


# using Toeplitz operator truncation
Ahr = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [0 1; -10*cos(t) -24-10*sin(t)],2pi));
psys = ps(Ahr,rand(2,1),rand(1,2))
@time ev2 = pspole(psys,50)
@test sort(ev) ≈ sort(real(ev2[sortperm(imag(ev2),by=abs)][1:2]))


# example Zhou, Hagiwara SCL 2002 period pi/2 and pi
# using Floquet based approach
At = PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi/2);
At2 = PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi);
psys = ps(At,rand(2,1),rand(1,2))
psys2 = ps(At2,rand(2,1),rand(1,2))
@time ev = pspole(psys; reltol = 1.e-10)
@test sort(real(ev)) ≈ sort([-1;-2]) && sort(imag(ev)) ≈ [2;2]
@time ev = pspole(psys, 500; reltol = 1.e-10)
@test sort(real(ev)) ≈ sort([-1;-2]) && sort(imag(ev)) ≈ [2;2]
@time ev2 = pspole(psys2; reltol = 1.e-10)
@test sort(ev2) ≈ sort([-1;-2])
@time ev2 = pspole(psys2,500; reltol = 1.e-10)
@test sort(ev2) ≈ sort([-1;-2])


# using Toeplitz operator truncation period pi/2
Ahr = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi/2));
psys = ps(Ahr,rand(2,0),rand(0,2))
ev3 = pspole(psys,5)
@test sort(real(ev3)) ≈ sort([-1;-2]) && sort(imag(ev3)) ≈ [2;2] 

# simple period
ev4 = pspole(psys,50)
@test sort(real(ev4)) ≈ sort([-1;-2]) && sort(imag(ev4)) ≈ [2;2] 

# double period
ev5 = pspole(psys,50; P = 2)
@test sort(real(ev5)) ≈ sort([-1;-2]) && norm(imag(ev5)) < 1.e-10 


# using Fourier series
Afun = FourierFunctionMatrix(Fun(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],Fourier(0..π/2)))
psys = ps(Afun,rand(2,1),rand(1,2))
ev3 = pspole(psys)
@test sort(real(ev3)) ≈ sort([-1;-2]) && sort(imag(ev3)) ≈ [2;2] 

# using Fourier series truncation period pi
ev4 = pspole(psys, P = 2)
@test sort(real(ev4)) ≈ sort([-1;-2]) && norm(imag(ev4)) < 1.e-10 


# using Toeplitz operator truncation period pi
Ahr2 = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi));
psys2 = ps(Ahr2,rand(2,0),rand(0,2))
ev3 = pspole(psys2,5)
@test sort(real(ev3)) ≈ sort([-1;-2]) && norm(imag(ev3)) < 1.e-10 

ev4 = pspole(psys2,50; P = 2)
@test sort(real(ev4)) ≈ sort([-1;-2]) && norm(imag(ev4)) < 1.e-10 

# constant system case
Q = convert(HarmonicArray,PeriodicFunctionMatrix([-1 0; 0 -2],pi/2));
psys = ps(Q,rand(2,0),rand(0,2))
ev1 = pspole(psys,5)
@test sort(real(ev1)) ≈ sort([-1;-2]) && norm(imag(ev1)) < 1.e-10

QF = convert(FourierFunctionMatrix,Q)
#psys = ps(QF,rand(2,0),rand(0,2)) # null dimensions not supported in ApproxFun
psys = ps(QF,rand(2,1),rand(1,2))
ev2 = pspole(psys,5)
@test sort(real(ev2)) ≈ sort([-1;-2]) && norm(imag(ev2)) < 1.e-10



# Floquet approach
A = PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi);
psys = ps(A,rand(2,1),rand(1,2))
@time ev = pspole(psys; reltol = 1.e-10)
@time ev = pspole(psys,50; reltol = 1.e-10)

# Fourier approach
Afun = FourierFunctionMatrix(Fun(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],Fourier(0..π)))
psys1 = ps(Afun,rand(2,1),rand(1,2))
ev1 = pspole(psys1)
@test sort(real(ev)) ≈ sort(real(ev1)) && norm(imag(ev1)) < 1.e-10

ev2 = pspole(psys1,5)
@test sort(real(ev)) ≈ sort(real(ev2)) && norm(imag(ev2)) < 1.e-10

ev3 = pspole(psys1,20)
@test sort(real(ev)) ≈ sort(real(ev3)) && norm(imag(ev3)) < 1.e-10


# # lossy Mathieu differential equation
# k = -1.; ξ = 0.05; β =0.2; ωh = 2; # unstable feedback
# k = -.5; ξ = 0.05; β =0.2; ωh = 2; # stable feedback
# Ahr = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [0 1; k*(1-2β*cos(ωh*t)) -2*ξ],2pi));
# Asym = convert(PeriodicSymbolicMatrix,Ahr); Asym.F
# N = 5; ev = eigvals(hr2bt(Ahr,N)-phasemat(Ahr,N))
# ev[sortperm(imag(ev),by=abs)][1:2]

# zeros example: infinite zeros present but not for the averaged system
a = PeriodicFunctionMatrix(t -> [sin(t)],2*pi);
b = PeriodicFunctionMatrix(t ->  [cos(t)], 2*pi); 
c = PeriodicFunctionMatrix(t ->  [cos(t)+sin(t)], 2*pi); 
d = [0];
psys = ps(HarmonicArray,a,b,c,d);
z = pszero(psys,atol=1.e-7)
@test z[1] == Inf
psys1 = ps(FourierFunctionMatrix,a,b,c,d)
z1 = pszero(psys1,30,atol=1.e-7)
@test any(isinf.(z1)) 

psys2 = ps(a,b,c,d); 
z2 = pszero(psys2,30,atol=1.e-7)
@test any(isinf.(z2)) 


# Bitanti-Colaneri's book p.26 
a = PeriodicFunctionMatrix(t -> [-1+sin(t) 0; 1-cos(t) -3],2*pi);
b = PeriodicFunctionMatrix(t ->  [-1-cos(t);2-sin(t)], 2*pi); 
c = [0 1]; d = [0];

psys = ps(a,b,c,d)
evref = pspole(psys,100)
@test sort(real(evref)) ≈ sort([-1,-3])  && norm(imag(evref)) < 1.e-10

# using Harmonic Array based computation
psyschr = ps(HarmonicArray,a,b,c,d)
ev = pspole(psyschr,20)
@test sort(real(evref)) ≈ sort(real(ev))  && norm(imag(ev)) < 1.e-10

Z = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [(-2+3*sin(t))/(2-sin(t))],2*pi))
ρ = real(Z.values[:,:,1])[1,1]

z = pszero(psyschr,atol=1.e-7)
@test z ≈ [ρ;Inf]


# using Fourier Function Matrix based computation
psysc = ps(FourierFunctionMatrix,a,b,c,d);
ev1 = pspole(psysc,30)
@test sort(real(evref)) ≈ sort(real(ev1))  && norm(imag(ev1)) < 1.e-10

z1 = pszero(psysc,atol=1.e-7)
@test minimum(abs.(z1.-ρ)) < 1.e-10  && any(isinf.(z1))

psys2 = ps(a,b,c,d); 
z2 = pszero(psys2,30,atol=1.e-7)
@test z2 ≈ [ρ;Inf]


# Zhou-Hagiwara Automatica 2002 
β = 0.5
a1 = PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi);
#γ(t) = mod(t,pi) < pi/2 ? sin(2*t) : 0 
γ = chop(Fun(t -> mod(t,pi) < pi/2 ? sin(2*t) : 0, Fourier(0..pi)),1.e-7);
#b1 = PeriodicFunctionMatrix(t ->  [0; 1-2*β*(mod(t,float(pi)) < pi/2 ? sin(2*t) : 0 )], pi); 
b1 = PeriodicFunctionMatrix(t ->  [0; 1-2*β*γ(t)], pi); 
c = [1 1]; d = [0];

# using Harmonic Array based lifting
psyschr = ps(HarmonicArray,a1,b1,c,d);
ev = pspole(psyschr)
@test sort([-1,-2]) ≈ sort(real(ev))  && norm(imag(ev)) < 1.e-10

z = pszero(psyschr,atol=1.e-7)
@test z ≈ [-3.5;Inf]


# using Fourier Function Matrix based lifting
psysc = ps(FourierFunctionMatrix,a1,b1,c,d);
ev = pspole(psysc)
@test sort([-1,-2]) ≈ sort(real(ev))  && norm(imag(ev)) < 1.e-10

z = pszero(psysc,30,atol=1.e-7)
@test z ≈ [-3.5;Inf]

psys2 = ps(a1,b1,c,d); 
z2 = pszero(psys2,30,atol=1.e-7)
@test z2 ≈  [-3.5;Inf]


# Zhou IEEE TAC 2009 
β = 0.5
a1 = PeriodicFunctionMatrix(t -> [-1-sin(2*t)^2 2-0.5*sin(4*t); -2-0.5*sin(4*t) -1-cos(2*t)^2],pi);
b1 = PeriodicFunctionMatrix(t ->  [0; 1-2*β*cos(2*t)], pi); 
c = [0 1]; d = [0];

# using Harmonic Array based lifting
psyschr = ps(HarmonicArray,a1,b1,c,d);
ev = pspole(psyschr)
@test sort([-1,-2]) ≈ sort(real(ev))  && norm(imag(ev)) < 1.e-10

z = pszero(psyschr,atol=1.e-7)
@test z ≈ [-1.5;Inf]

psysc = ps(FourierFunctionMatrix,a1,b1,c,d);
ev = pspole(psysc)
@test sort([-1,-2]) ≈ sort(real(ev))  && norm(imag(ev)) < 1.e-10

z = pszero(psysc,30,atol=1.e-7)
@test z ≈ [-1.5;Inf]

psys2 = ps(a1,b1,c,d); 
z2 = pszero(psys2,30,atol=1.e-7)
@test z2 ≈  [-1.5;Inf]

# # Ziegler's column

# β = 2pi; η =0.5; λ = 0.5; 
# M = [1 3/8; 3/2 1];
# K0 = [3/8 3/16; -3/4 3/4]; K1 = λ*[-1 η; 0 4η-4];
# A = PeriodicFunctionMatrix(τ -> [eye(2) zeros(2,2); zeros(2,2) -M\(K0+cos(β*τ).*K1)],2*pi/β)

# # Mathieu equation: determination of minimum eigenvalue
# q = 2; a = 2; a = -1.513956885056448; a = -1
# Ahr = convert(HarmonicArray,PeriodicFunctionMatrix(t -> [0 1; a-2*q*cos(2*t) 0],pi));
# Asym = convert(PeriodicSymbolicMatrix,Ahr); Asym.F
# ev = psceig(Ahr)
# N = 20; ev = eigvals(hr2bt(Ahr,N)-phasemat(Ahr,N))
# ev[sortperm(imag(ev),by=abs)][1:2]

# N = 5; ev1 = eigvals(hr2bt1(Ahr,N,2)-phasemat1(Ahr,N,2))
# ev1[sortperm(imag(ev1),by=abs)][1:2]

# discrete systems
Ad = PeriodicMatrix([[8. 0; 0 0], [1 1;1 1], [0 1; 1 0]], 6, nperiod = 2);
Bd = PeriodicMatrix( [[ 1; 0 ], [ 1; 1]] ,2);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 0]] ,2);
Dd = PeriodicMatrix( [[ 1 ]], 1);
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd); 
ev = pspole(psys)
ev1 = pspole(psys,7)
@test sort(ev) ≈ sort([2,0]) ≈ sort(ev1) ≈ sort(gpole(ps2ls(psys))).^(1/6)

z = pszero(psys)
z1 = gzero(ps2ls(psys))
@test all(abs.(z[isfinite.(z)].^6-z1[isfinite.(z1)]) .< 1.e-8) && length(z[isinf.(z)]) == length(z1[isinf.(z1)])


Ad = PeriodicMatrix([[4. 0], [1;1], [2 0;0 1]],3);
Bd = PeriodicMatrix( [[ 1 ], [ 1; 1], [1;2]] ,3);
Cd = PeriodicMatrix( [[ 1 1 ], [ 1 ], [1 0]] ,3);
Dd = PeriodicMatrix( [[ 1 ]], 1); 
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd);
ev = pspole(psys)
@test sort(ev) ≈ sort([2,0])
ev1 = pspole(psys,2)
@test ev1 ≈ [2] ≈ gpole(ps2ls(psys,2)).^(1/3)

z = pszero(psys)
z1 = gzero(ps2ls(psys))
@test all(abs.(z[isfinite.(z)].^3-z1[isfinite.(z1)]) .< 1.e-8) && length(z[isinf.(z)]) == length(z1[isinf.(z1)])

Ad = PeriodicMatrix([[1. 2], [7;8]],2);
Bd = PeriodicMatrix( [[ 3 ], [ 9; 10]] ,2);
Cd = PeriodicMatrix( [[ 4 5 ], [ 11 ]] ,2);
Dd = PeriodicMatrix( [[ 0 ]], 1); 
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd);

z = pszero(psys)
z1 = gzero(ps2ls(psys))
@test all(abs.(z[isfinite.(z)].^2-z1[isfinite.(z1)]) .< 1.e-8) && length(z[isinf.(z)]) == length(z1[isinf.(z1)])

z = pszero(psys,2)
z1 = gzero(ps2ls(psys,2))
@test all(abs.(z[isfinite.(z)].^2-z1[isfinite.(z1)]) .< 1.e-8) && length(z[isinf.(z)]) == length(z1[isinf.(z1)])

psysa = convert(PeriodicStateSpace{PeriodicArray},psys)
za = pszero(psysa)
za1 = gzero(ps2ls(psysa))
@test all(abs.(za[isfinite.(za)].^2-za1[isfinite.(za1)]) .< 1.e-8) && length(za[isinf.(za)]) == length(za1[isinf.(za1)])

za = pszero(psysa,2)
za1 = gzero(ps2ls(psysa,2))
@test all(abs.(za[isfinite.(za)].^2-za1[isfinite.(za1)]) .< 1.e-8) && length(za[isinf.(za)]) == length(za1[isinf.(za1)])


# periodic array representation
Ad = PeriodicArray(rand(Float32,2,2,10),10);
Bd = PeriodicArray(rand(2,1,2),2);
Cd = PeriodicArray(rand(1,2,3),3);
Dd = PeriodicArray(zeros(1,1,1), 1);
psys = PeriodicStateSpace(Ad,Bd,Cd,Dd); 
ev = pspole(psys)
ev1 = pspole(psys; fast = true)
@test sort(real(ev)) ≈ sort(real(ev1)) && norm(sort(imag(ev))-sort(imag(ev1))) < 1.e-7

z = pszero(psys,atol=1.e-7)
z1 = gzero(ps2ls(psys),atol=1.e-7)
@test all(abs.(z[isfinite.(z)].^30-z1[isfinite.(z1)]) .< 1.e-8) && length(z[isinf.(z)]) == length(z1[isinf.(z1)])


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
@time evc = pspole(psysc)
@test all(abs.(real(evc)) .< 1.e-10)

zc = pszero(psysc)
@test all(isinf.(zc))


K = 120;
@time psys = psc2d(psysc,period/K,reltol = 1.e-10);
@time evd = pspole(psys)
@test all(abs.(evd) .≈ 1)

zd = pszero(psys,atol=1.e-7)
@test all(isinf.(zd))

end # test

end # module