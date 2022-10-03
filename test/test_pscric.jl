module Test_pscric

using PeriodicSystems
using DescriptorSystems
using MatrixEquations
using Symbolics
using Test
using LinearAlgebra
using ApproxFun
using Symbolics
#using JLD

println("Test_pcric")

@testset "pcric" begin

# example Johanson et al. 2007
A = [1 0.5; 3 5]; B = [3;1;;]; Q = [1. 0;0 1]; R = [1.;;]
period = π; 
ω = 2. ;

Xref, EVALSref, Fref = arec(A,B,R,Q); 

Ac = PeriodicTimeSeriesMatrix(A,period)
Bc = PeriodicTimeSeriesMatrix(B,period)
@time Xc, EVALSc, Fc = prcric(Ac, Bc, R, Q)
@test iszero(Xc-Xref) && iszero(Fc-Fref) && iszero(EVALSc-EVALSref)


# @variables t
# P = PeriodicSymbolicMatrix([cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); PM = PeriodicSymbolicMatrix

P1 = PeriodicFunctionMatrix(t->[cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); PM = PeriodicFunctionMatrix

#P = convert(HarmonicArray,P); PM = HarmonicArray

#P = convert(FourierFunctionMatrix,P); PM = FourierFunctionMatrix
PM = HarmonicArray
PM = PeriodicTimeSeriesMatrix


for PM in (PeriodicFunctionMatrix, HarmonicArray, PeriodicSymbolicMatrix, FourierFunctionMatrix, PeriodicTimeSeriesMatrix)
    println("type = $PM")
    N = PM == PeriodicTimeSeriesMatrix ? 128 : 200
    P = convert(PM,P1);

    Ap = derivative(P)*inv(P)+P*A*inv(P);
    Bp = P*B
    Qp = inv(P)'*Q*inv(P); Qp = (Qp+Qp')/2
    Rp = PM(R, Ap.period)
    Xp = inv(P)'*Xref*inv(P)
    @test Ap'*Xp+Xp*Ap+Qp-Xp*Bp*Bp'*Xp ≈ -derivative(Xp)
    Fp = Bp'*Xp
    @test norm(sort(real(psceig(Ap-Bp*Fp))) - sort(EVALSref)) < 1.e-2
    
    solver = "symplectic"
    for solver in ("non-stiff", "stiff", "symplectic", "linear", "noidea")
        println("solver = $solver")
        @time X, EVALS, F = prcric(Ap, Bp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = true) 
        Errx = norm(X-Xp)/norm(Xp); Errf = norm(F-Fp)/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
        @time X, EVALS, F = prcric(Ap, Bp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = false) 
        Errx = norm(X-Xp)/norm(Xp); Errf = norm(F-Fp)/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
    end
end   

A = [1 0.5; 3 5]; C = [3 1]; Q = [1. 0;0 1]; R = [1.;;]
period = π; 
ω = 2. ;

Xref, EVALSref, Fref = arec(A', C', R, Q); Fref = copy(Fref')
@test norm(A*Xref+Xref*A'-Xref*C'*inv(R)*C*Xref +Q) < 1.e-7

Ac = PeriodicTimeSeriesMatrix(A,period)
Cc = PeriodicTimeSeriesMatrix(C,period)
@time Xc, EVALSc, Fc = pfcric(Ac, Cc, R, Q)
@test iszero(Xc-Xref) && iszero(Fc-Fref) && iszero(EVALSc-EVALSref)


P1 = PeriodicFunctionMatrix(t->[cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); 

PM = HarmonicArray
PM = PeriodicTimeSeriesMatrix
PM = PeriodicSymbolicMatrix
for PM in (PeriodicFunctionMatrix, HarmonicArray, PeriodicSymbolicMatrix, FourierFunctionMatrix, PeriodicTimeSeriesMatrix)
    println("type = $PM")
    N = PM == PeriodicTimeSeriesMatrix ? 128 : 200
    P = convert(PM,P1);

    Ap = derivative(P)*inv(P)+P*A*inv(P);
    Cp = C*inv(P)
    Qp = P*Q*P'; Qp = (Qp+Qp')/2
    Rp = PM(R, Ap.period)
    Xp = P*Xref*P'
    Gp = Cp'*inv(Rp)*Cp
    @test Ap*Xp+Xp*Ap'+Qp-Xp*Gp*Xp ≈ derivative(Xp)
    Fp = Xp*Cp'
    @test norm(sort(real(psceig(Ap-Fp*Cp))) - sort(EVALSref)) < 1.e-2

    
    solver = "symplectic" 
    solver = "stiff"
    for solver in ("non-stiff", "stiff", "symplectic", "linear", "noidea")
        println("solver = $solver")
        # @time X1, EVALS1, F1 = prcric(Ap1, Cp', Rp1, Qp1; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = true) 
        # @test norm(Ap*X1+X1*Ap'+Qp-X1*Gp*X1 -derivative(X1)) < 1.e-7
        @time X, EVALS, F = pfcric(Ap, Cp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = true) 
        #@test norm(Ap*X2+X2*Ap'+Qp-X2*Gp*X2 -derivative(X2)) < 1.e-6
        Errx = norm(X-Xp)/norm(Xp); Errf = norm(F-Fp)/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
        @time X, EVALS, F = pfcric(Ap, Cp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = false) 
        #@test norm(Ap*X+X*Ap'+Qp-X*Gp*X -derivative(X)) < 1.e-6 ##&& norm(sort(real(psceig(Ap-F*Cp))) - sort(EVALSref)) < 1.e-2
        Errx = norm(X-Xp)/norm(Xp); Errf = norm(F-Fp)/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
    end
end   

# random examples

n = 20; m = 5; nh = 2; period = π; 

A = pmrand(n, n, period; nh)
B = pmrand(n, m, period; nh)
Q = collect(Float64.(I(n))); R = collect(Float64.(I(m))); 
ev = psceig(A,100)
@time X, EVALS, F = prcric(A, B, R, Q; K = 100, solver = "nonstiff", reltol = 1.e-10, abstol = 1.e-10, fast = true); 

@test all(real(psceig(A-B*F,100)) .< 0)

println(norm(A'*X+X*A+Q-X*B*B'*X +derivative(X))/norm(X))

@time X, EVALS, F = prcric(A, B, R, Q; K = 100, solver = "nonstiff", reltol = 1.e-10, abstol = 1.e-10, fast = false); 

@test all(real(psceig(A-B*F,100)) .< 0)

println(norm(A'*X+X*A+Q-X*B*B'*X +derivative(X))/norm(X))


# Pitelkau's example - singular Lyapunov equations
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
q = [2 0 0 0; 0 1 0 0; 0 0 0 0;0 0 0 0]; r = [1.e-11;;];

ev = psceig(HarmonicArray(a,2pi))
PM = HarmonicArray
PM = PeriodicFunctionMatrix
for PM in (PeriodicFunctionMatrix, HarmonicArray)
    println("PM = $PM")
    psysc = ps(a,convert(PM,PeriodicFunctionMatrix(b,period)),c,d);

    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 200, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true); 
    clev = psceig(psysc.A-psysc.B*F,500)
    println("EVALS = $EVALS, clev = $clev")
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    @test norm(psysc.A'*X+X*psysc.A+q-X*psysc.B*inv(r)*psysc.B'*X +derivative(X))/norm(X) < 1.e-7 

    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 200, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = false ); 
    clev = psceig(psysc.A-psysc.B*F,500)
    println("EVALS = $EVALS, clev = $clev")
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    @test norm(psysc.A'*X+X*psysc.A+q-X*psysc.B*inv(r)*psysc.B'*X +derivative(X))/norm(X) < 1.e-7 


    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 200, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = true); 
    clev = psceig(psysc.A-psysc.B*F,500)
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-1 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-1 
    @test norm(psysc.A'*X+X*psysc.A+q-X*psysc.B*inv(r)*psysc.B'*X +derivative(X))/norm(X) < 1.e-7 
end 
end

end # module

