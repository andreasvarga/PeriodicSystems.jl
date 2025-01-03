"""
     ps2frls(psysc::PeriodicStateSpace, N) -> sys::DescriptorStateSpace 

Build the real frequency-lifted representation of a continuous-time periodic system.

For a continuos-time periodic system `psysc = (A(t),B(t),C(t),D(t))`, the real 
LTI state-space representation `sys = (At-Nt,Bt,Ct,Dt)` is built, where `At`, `Bt`, `Ct` and `Dt` 
are truncated block Toeplitz matrices and `Nt` is a block diagonal matrix. 
`N` is the number of selected harmonic components in the Fourier series of system matrices. 

_Note:_ This is an experimental implementation based on the operator representation of periodic matrices
in the [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) package. 
"""
function PeriodicSystems.ps2frls(psysc::PeriodicStateSpace{PM}, N::Int; P::Int= 1) where {T,PM <: AbstractPeriodicArray{:c,T}}
    psyscfr = typeof(psysc) <: PeriodicStateSpace{FourierFunctionMatrix} ? psysc :
              convert(PeriodicStateSpace{FourierFunctionMatrix},psysc)
    N >= 0 || error("number of selected harmonics must be nonnegative, got $N")   
    (Af, Bf, Cf, Df) = P == 1 ? (psyscfr.A, psyscfr.B, psyscfr.C, psyscfr.D) :
                                (FourierFunctionMatrix(Fun(t -> psyscfr.A.M(t),Fourier(0..P*psyscfr.A.period))), 
                                 FourierFunctionMatrix(Fun(t -> psyscfr.B.M(t),Fourier(0..P*psyscfr.B.period))), 
                                 FourierFunctionMatrix(Fun(t -> psyscfr.C.M(t),Fourier(0..P*psyscfr.C.period))),
                                 FourierFunctionMatrix(Fun(t -> psyscfr.D.M(t),Fourier(0..P*psyscfr.D.period))))
     
    n, m = size(Bf); p = size(Cf,1);
    D = Derivative(domain(Af.M))
    ND = DiagDerOp(D,n)
    Aop = Af.M - ND
    Cop = Multiplication(Cf.M,domainspace(ND))
    sdu = domainspace(DiagDerOp(0*D,m))
    Bop = Multiplication(Bf.M,sdu)
    Dop = Multiplication(Df.M,sdu)
    Ntx = 2*n*(2*N+1)
    Ntu = m*(2*N+1)
    Nty = p*(2*N+1)
    sys = dss(Matrix(Aop[1:Ntx,1:Ntx]), Matrix(Bop[1:Ntx,1:Ntu]), Matrix(Cop[1:Nty,1:Ntx]), Matrix(Dop[1:Nty,1:Ntu]))
    return sys  
end      

function DiagDerOp(D::Union{ApproxFunBase.DerivativeWrapper,ApproxFunBase.ConstantTimesOperator}, n::Int) 
    Z = tuple(D,ntuple(n->0I,n-1)...)
    for i = 2:n
        Z1 = tuple(ntuple(n->0I,i-1)...,D,ntuple(n->0I,n-i)...)
        Z = tuple(Z...,Z1...)
    end
    return hvcat(n,Z...)
end

