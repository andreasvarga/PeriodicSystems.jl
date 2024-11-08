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

# Ac = PeriodicTimeSeriesMatrix(A,period)
# Bc = PeriodicTimeSeriesMatrix(B,period)
# @time Xc, EVALSc, Fc = prcric(Ac, Bc, R, Q)
# @test iszero(Xc-Xref) && iszero(Fc-Fref) && iszero(EVALSc-EVALSref)


# @variables t
# P = PeriodicSymbolicMatrix([cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); PM = PeriodicSymbolicMatrix

P1 = PeriodicFunctionMatrix(t->[cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); 
P1dot = PeriodicFunctionMatrix(t->[-ω*sin(t*ω)   ω*cos(t*ω); -ω*cos(t*ω)  -ω*sin(t*ω)],period); 
# PM = PeriodicFunctionMatrix
# PM = PeriodicTimeSeriesMatrix

#P = convert(HarmonicArray,P); 
PM = HarmonicArray

#P = convert(FourierFunctionMatrix,P); PM = FourierFunctionMatrix
# PM = PeriodicSymbolicMatrix

for PM in (PeriodicFunctionMatrix, HarmonicArray, PeriodicSymbolicMatrix, FourierFunctionMatrix, PeriodicTimeSeriesMatrix)
    println("type = $PM")
    N = PM == PeriodicTimeSeriesMatrix ? 128 : 200
    P = convert(PM,P1);
    Pdot = convert(PM,P1dot);

    #Ap = pmderiv(P)*inv(P)+P*A*inv(P);
    Ap = Pdot*inv(P)+P*A*inv(P);
    Bp = P*B
    Qp = inv(P)'*Q*inv(P); Qp = (Qp+Qp')/2
    Rp = PM(R, Ap.period)
    Xp = inv(P)'*Xref*inv(P)
    Fp = Bp'*Xp
    # if PM == PeriodicTimeSeriesMatrix 
    #     Xp = inv(P1)'*Xref*inv(P1)
    #     Fp = (P1*B)'*Xp
    # end
    if PM == PeriodicTimeSeriesMatrix 
       @test norm(Ap'*Xp+Xp*Ap+Qp-Xp*Bp*Bp'*Xp + pmderiv(Xp)) < 1.e-5*norm(Xp)
    else
       @test Ap'*Xp+Xp*Ap+Qp-Xp*Bp*Bp'*Xp ≈ -pmderiv(Xp)
       @test norm(sort(real(psceig(Ap-Bp*Fp,100))) - sort(EVALSref)) < 1.e-6  
    end
    
    solver = "symplectic"
    #N = length(Xp.values)
    ti = collect((0:N-1)*Xp.period/N)*(1+eps(10.))
    for solver in ("non-stiff", "stiff", "symplectic", "linear", "noidea")
        println("solver = $solver")
        @time X, EVALS, F = prcric(Ap, Bp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = true) #error
        Errx = norm(X.(ti)-Xp.(ti))/norm(Xp); Errf = norm(F.(ti)-Fp.(ti))/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
        @time X, EVALS, F = prcric(Ap, Bp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = false) 
        Errx = norm(X.(ti)-Xp.(ti))/norm(Xp); Errf = norm(F.(ti)-Fp.(ti))/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
    end
end   

A = [1 0.5; 3 5]; C = [3 1]; Q = [1. 0;0 1]; R = [1.;;]
period = π; 
ω = 2. ;

Xref, EVALSref, Fref = arec(A', C', R, Q); Fref = copy(Fref')
@test norm(A*Xref+Xref*A'-Xref*C'*inv(R)*C*Xref +Q) < 1.e-7

# Ac = PeriodicTimeSeriesMatrix(A,period)
# Cc = PeriodicTimeSeriesMatrix(C,period)
# @time Xc, EVALSc, Fc = pfcric(Ac, Cc, R, Q)
# @test iszero(Xc-Xref) && iszero(Fc-Fref) && iszero(EVALSc-EVALSref) 


P1 = PeriodicFunctionMatrix(t->[cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)],period); 
P1dot = PeriodicFunctionMatrix(t->[-ω*sin(t*ω)   ω*cos(t*ω); -ω*cos(t*ω)  -ω*sin(t*ω)],period); 

# PM = HarmonicArray
# PM = PeriodicTimeSeriesMatrix
# PM = PeriodicSymbolicMatrix
PM = FourierFunctionMatrix

for PM in (PeriodicFunctionMatrix, HarmonicArray, PeriodicSymbolicMatrix, FourierFunctionMatrix, PeriodicTimeSeriesMatrix)
    println("type = $PM")
    N = PM == PeriodicTimeSeriesMatrix ? 128 : 200
    P = convert(PM,P1);
    Pdot = convert(PM,P1dot);

    Ap = Pdot*inv(P)+P*A*inv(P);
    Cp = C*inv(P)
    Qp = P*Q*P'; Qp = (Qp+Qp')/2
    Rp = PM(R, Ap.period)
    Xp = P*Xref*P'
    Gp = Cp'*inv(Rp)*Cp
    Fp = Xp*Cp'
    if PM == PeriodicTimeSeriesMatrix 
        @test norm(Ap*Xp+Xp*Ap'+Qp-Xp*Gp*Xp - pmderiv(Xp)) < 1.e-5*norm(Xp)
    else
        @test Ap*Xp+Xp*Ap'+Qp-Xp*Gp*Xp ≈ pmderiv(Xp)
        @test norm(sort(real(psceig(Ap-Fp*Cp,100))) - sort(EVALSref)) < 1.e-6   
    end
 
     
    solver = "symplectic" 
    #solver = "stiff"
    ti = collect((0:N-1)*Xp.period/N)*(1+eps(10.))
    for solver in ("non-stiff", "stiff", "symplectic", "linear", "noidea")
        println("solver = $solver")
        @time X, EVALS, F = pfcric(Ap, Cp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = true) 
        Errx = norm(X.(ti)-Xp.(ti))/norm(Xp); Errf = norm(F.(ti)-Fp.(ti))/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
        @time X, EVALS, F = pfcric(Ap, Cp, Rp, Qp; K = N, solver, reltol = 1.e-10, abstol = 1.e-10, fast = false) 
        #@test norm(Ap*X+X*Ap'+Qp-X*Gp*X -pmderiv(X)) < 1.e-6 ##&& norm(sort(real(psceig(Ap-F*Cp))) - sort(EVALSref)) < 1.e-2
        Errx = norm(X.(ti)-Xp.(ti))/norm(Xp); Errf = norm(F.(ti)-Fp.(ti))/norm(Fp)
        println("Errx = $Errx Errf = $Errf")
        @test Errx < 1.e-7 && Errf < 1.e-6 && norm(sort(real(EVALS)) - sort(EVALSref)) < 1.e-2
    end
end   

## this seems to take infinite time
# random examples

n = 20; m = 5; nh = 2; period = π; 
#n = 5; m = 5; nh = 2; period = π; 
ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]*0.5


A = pmrand(n, n, period; nh)
B = pmrand(n, m, period; nh)
Q = collect(Float64.(I(n))); R = collect(Float64.(I(m))); Qt = HarmonicArray(Q,pi)
ev = psceig(A,100)
@time X, EVALS, F = prcric(A, B, R, Q; K = 100, solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10, fast = true); 

@test all(real(psceig(A-B*F,100)) .< 0)

Xdot = pmderiv(X); 
@test norm(A'.(ts).*X.(ts).+X.(ts).*A.(ts).+Qt.(ts).-X.(ts).*B.(ts).*B'.(ts).*X.(ts) .+ Xdot.(ts),Inf)/norm(X.(ts),Inf) < 1.e-7

@time X, EVALS, F = prcric(A, B, R, Q; K = 100, solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10, fast = false); 

@test all(real(psceig(A-B*F,100)) .< 0)

Xdot = pmderiv(X); 
@test norm(A'.(ts).*X.(ts).+X.(ts).*A.(ts).+Qt.(ts).-X.(ts).*B.(ts).*B'.(ts).*X.(ts) .+ Xdot.(ts),Inf)/norm(X.(ts),Inf) < 1.e-7


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
ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]/ω


for PM in (PeriodicFunctionMatrix, HarmonicArray)
    println("PM = $PM")
    psysc = ps(a,convert(PM,PeriodicFunctionMatrix(b,period)),c,d);
    Qt = PM(q,period)
    Rit = PM(inv(r),period)

    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true,intpol = true); 
    #@time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true); 
    clev = psceig(psysc.A-psysc.B*F,500)
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    @test norm(psysc.A'.(ts).*X.(ts).+X.(ts).*psysc.A.(ts).+Qt.(ts).-X.(ts).*psysc.B.(ts).*Rit.(ts).*psysc.B'.(ts).*X.(ts).+pmderiv(X).(ts))/norm(X.(ts)) < 1.e-5 

    # this test covers the experimental code provided in PeriodicSchurDecompositions package and occasionally fails
    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = false,intpol = true ); 
    #@time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = false ); 
    clev = psceig(psysc.A-psysc.B*F,500)
    println("EVALS = $EVALS, clev = $clev")
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-7 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-7 
    @test norm(psysc.A'.(ts).*X.(ts).+X.(ts).*psysc.A.(ts).+Qt.(ts).-X.(ts).*psysc.B.(ts).*Rit.(ts).*psysc.B'.(ts).*X.(ts).+pmderiv(X).(ts))/norm(X.(ts)) < 1.e-5 


    @time X, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = true,intpol = true); 
    clev = psceig(psysc.A-psysc.B*F,500)
    @test norm(sort(real(clev)) - sort(real(EVALS))) < 1.e-1 && norm(sort(imag(clev)) - sort(imag(EVALS))) < 1.e-1 
     @test norm(psysc.A'.(ts).*X.(ts).+X.(ts).*psysc.A.(ts).+Qt.(ts).-X.(ts).*psysc.B.(ts).*Rit.(ts).*psysc.B'.(ts).*X.(ts).+pmderiv(X).(ts))/norm(X.(ts)) < 1.e-5 
end 

# psysc = ps(a,convert(PM,PeriodicFunctionMatrix(b,period)),c,d);

# X1, EVALS, F = prcric(psysc.A, psysc.B, r, q; K = 100, solver = "symplectic", reltol = 1.e-10, abstol = 1.e-10, fast = false, PSD_SLICOT = true);
# Xts, ev = pgcric(psysc.A, psysc.B*inv(r)*psysc.B', HarmonicArray(q,psysc.period), 100; adj = true);
# @test X1(0) ≈ Xts(0) && EVALS ≈ ev 

# T = psysc.period/100
# @test X1(T) ≈ Xts(T)
# t = T*100*rand();
# @test convert(HarmonicArray,Xts)(t) ≈ X1(t)
# Xt = PeriodicFunctionMatrix(t->PeriodicSystems.tvcric_eval(t,Xts,psysc.A, psysc.B*inv(r)*psysc.B', HarmonicArray(q,2pi); adj = true, solver = "symplectic", reltol = 1e-10, abstol = 1e-10),Xts.period)
# @test Xt(t) ≈ X1(t)


end  # test

end # module

