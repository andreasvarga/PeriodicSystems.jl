"""
     ps2ls(psys::PeriodicStateSpace; kstart, ss = false, cyclic = false) -> sys::DescriptorStateSpace 

Build the discrete-time lifted LTI system equivalent to a discrete-time periodic system. 

For a discrete-time periodic system `psys = (A(t),B(t),C(t),D(t))` with period `T` and sample time `Ts`, 
the equivalent stacked (see [1]) LTI descriptor state-space representation 
`sys = (A-λE,B,C,D)` is built, with the input, state and output vectors defined over time intervals of length `T` (instead `Ts`).  
The keyword argument `kstart = k`, specifies a desired time `k` to start the sequence of periodic matrices.
If `ss = true` (default: `ss = false`), then all non-dynamic modes are elliminated and 
a standard state-space realization (with `E = I`) is determined, which corresponds to the lifting techniques of [2], where
only the input and output vectors are defined over time intervals of length `T`. 

If cyclic = true, the cyclic reformulation of [3] is used to build a lifted standard system with 
the input, state and output vectors defined over time intervals of length `T` (instead `Ts`).

_Note:_ The determination of the standard lifted representation involves forming matrix products 
(e.g., by explicitly forming the monodromy matrix) and therefore is potentially less suited for numerical computations.  

_References_

[1] O. M. Grasselli and S. Longhi. Finite zero structure of linear periodic discrete-time systems. 
    Int. J. Systems Sci. 22, 1785–1806,  1991.

[2] R. A. Meyer and C. S. Burrus. A unified analysis of multirate and periodically time-varying
    digital filters”, IEEE Transactions on Circuits and Systems, Vol. 22, pp. 162–167, 1975.

[3] D. S. Flamm. A new shift-invariant representation for periodic systems, 
    Systems and Control Letters, Vol. 17, No. 1, pp. 9–14, 1991.
"""
function ps2ls(psys::PeriodicStateSpace{PeriodicMatrix{:d,T}}; kstart::Int = 1, ss::Bool = false, cyclic::Bool = false) where {T}
    pa = psys.A.dperiod
    na = psys.A.nperiod
    ndx, nx = size(psys.A)
    Ts = psys.A.Ts
    K = pa*na  # number of blocks
    N = sum(nx)*na  # dimension of lifted state vector 
    pb = psys.B.dperiod
    nb = psys.B.nperiod
    pc = psys.C.dperiod
    nc = psys.C.nperiod
    pd = psys.D.dperiod
    nd = psys.D.nperiod
    m = psys.nu[1]
    M = K*m
    p = psys.ny[1]
    P = K*p
    B = zeros(T, N, M) 
    jfin = mod(kstart+pb-2,pb)+1
    n = ndx[jfin]
    copyto!(view(B,1:n,M-m+1:M),psys.B.M[jfin])
    i1 = n+1
    j1 = 1
    pbc = pb
    for i = 1:nb
        k = kstart
        i == nb && (pbc = pb-1)
        for j = 1:pbc
            jj = mod(k-1,pb)+1
            i2 = i1+ndx[jj]-1
            j2 = j1+m-1
            copyto!(view(B,i1:i2,j1:j2),psys.B.M[jj])
            k += 1
            i1 = i2+1
            j1 = j2+1 
        end
    end
    C = zeros(T, P, N) 
    i1 = 1
    j1 = 1
    for i = 1:nc
        k = kstart
        for j = 1:pc
            jj = mod(k-1,pc)+1
            i2 = i1+p-1
            j2 = j1+nx[jj]-1
            copyto!(view(C,i1:i2,j1:j2),psys.C.M[jj])
            k += 1
            i1 = i2+1
            j1 = j2+1 
        end
    end
    D = zeros(T, P, M) 
    i1 = 1
    j1 = 1
    for i = 1:nd
        k = kstart
        for j = 1:pd
            i2 = i1+p-1
            j2 = j1+m-1
            copyto!(view(D,i1:i2,j1:j2),psys.D.M[mod(k-1,pd)+1])
            k += 1
            i1 = i2+1
            j1 = j2+1 
        end
    end
    A = zeros(T, N, N) 
    jafin = mod(kstart+pa-2,pa)+1
    nf = nx[jafin]   
    copyto!(view(A,1:n,N-nf+1:N),psys.A.M[jafin])
    i1 = n+1
    j1 = 1
    pac = pa
    U = cyclic ? 0I : -I
    for i = 1:na
        k = kstart
        i == na && (pac = pa-1)
        for j = 1:pac
            jj = mod(k-1,pa)+1
            jj1 = mod(k,pa)+1
            i2 = i1+ndx[jj]-1
            j2 = j1+nx[jj]-1
            copyto!(view(A,i1:i2,j1:j2),psys.A.M[jj])
            copyto!(view(A,i1:i2,j2+1:j2+nx[jj1]),U)
            k += 1
            i1 = i2+1
            j1 = j2+1 
        end
    end
    cyclic && (return dss(A, B, C, D; Ts))
    if ss 
       # generate the standard lifted system using residualization formulas for E = [I 0; 0 0]
       i1 = 1:n; i2 = n+1:N
       A11 = view(A,i1,i1)
       A12 = view(A,i1,i2)
       A21 = view(A,i2,i1)
       A22 = LowerTriangular(view(A,i2,i2))
       B1 = view(B,i1,:)
       B2 = view(B,i2,:)
       C1 = view(C,:,i1)
       C2 = view(C,:,i2)
       # make A22 = I
       ldiv!(A22,A21)
       ldiv!(A22,B2)
       # apply simplified residualization formulas
       mul!(D, C2, B2, -1, 1)
       mul!(B1, A12, B2, -1, 1)
       mul!(C1, C2, A21, -1, 1)
       mul!(A11, A12, A21, -1, 1)
       return dss(A11, B1, C1, D; Ts)
    else
       # generate the stacked lifted descriptor system of (Grasselli and Longhi, 1991)
       E = zeros(T, N, N) 
       copyto!(view(E,1:n,1:n),I)
       return dss(A, E, B, C, D; Ts)
    end
end
ps2ls(psys::PeriodicStateSpace{PeriodicArray{:d,T}}; kwargs...) where {T} = 
      ps2ls(convert(PeriodicStateSpace{PeriodicMatrix},psys); kwargs...)

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
function ps2frls(psysc::PeriodicStateSpace{PM}, N::Int) where {T,PM <: AbstractPeriodicArray{:c,T}}
    psyscfr = typeof(psysc) <: PeriodicStateSpace{FourierFunctionMatrix} ? psysc :
             convert(PeriodicStateSpace{FourierFunctionMatrix},psysc)
    N >= 0 || error("number of selected harmonics must be nonnegative, got $N")         
    n, m = size(psyscfr.B); p = size(psyscfr.C,1);
    D = Derivative(domain(psyscfr.A.M))
    ND = DiagDerOp(D,n)
    Aop = psyscfr.A.M - ND
    Cop = Multiplication(psyscfr.C.M,domainspace(ND))
    sdu = domainspace(DiagDerOp(0*D,m))
    Bop = Multiplication(psyscfr.B.M,sdu)
    Dop = Multiplication(psyscfr.D.M,sdu)
    Ntx = 2*n*(2*N+1)
    Ntu = m*(2*N+1)
    Nty = p*(2*N+1)
    sys = dss(Matrix(Aop[1:Ntx,1:Ntx]), Matrix(Bop[1:Ntx,1:Ntu]), Matrix(Cop[1:Nty,1:Ntx]), Matrix(Dop[1:Nty,1:Ntu]))
    return sys  
end      
"""
     ps2fls(psysc::PeriodicStateSpace, N; P) -> sys::DescriptorStateSpace 

Build the frequency-lifted representation of a continuous-time periodic system.

For a continuos-time periodic system `psysc = (A(t),B(t),C(t),D(t))`, the (complex) 
LTI state-space representation `sys = (At-Nt,Bt,Ct,Dt)` is built, where `At`, `Bt`, `Ct` and `Dt` 
are truncated block Toeplitz matrices and `Nt` is a block diagonal matrix (see [1] or [2]). 
`N` is the number of selected harmonic components in the Fourier series of system matrices 
and the keyword parameter `P` is the number of full periods to be considered (default: `P = 1`). 

_References_

[1] N. M. Wereley. Analysis and control of linear periodically time varying systems. 
    Ph.D. thesis, Department of Aeronautics and Astronautics, MIT, 1990.

[2] S. Bittanti and P. Colaneri. Periodic Systems : Filtering and Fontrol.
    Springer-Verlag London, 2009. 
"""
function ps2fls(psysc::PeriodicStateSpace{PM}, N::Int; P::Int= 1) where {T,PM <: AbstractPeriodicArray{:c,T}}
    psyshr = typeof(psysc) <: PeriodicStateSpace{HarmonicArray} ? psysc :
             convert(PeriodicStateSpace{HarmonicArray},psysc)
    N >= 0 || error("number of selected harmonics must be nonnegative, got $N")         
    P >= 1 || error("period multiplicator must be at least 1, got $P")         
    (PA, PB, PC, PD ) = (psyshr.A.nperiod, psyshr.B.nperiod, psyshr.C.nperiod, psyshr.D.nperiod)
    K = lcm(PA,PB,PC,PD)
    #println("K = $K PA = $PA PB = $PB PC = $PC PD = $PD")
     # sys = dss(hr2btupd(psyshr.A, N*div(K,PA); P = P*PA), hr2bt(psyshr.B,N*div(K,PB); P = P*PB),
    #           hr2bt(psyshr.C,N*div(K,PC); P = P*PC),hr2bt(psyshr.D,N*div(K,PD); P = P*PD)) 
    # sys = dss(hr2btupd(psyshr.A, N; P = P*PA, nperiod = div(K,PA)), hr2bt(psyshr.B, N; P = P*PB, nperiod = div(K,PB)),
    #           hr2bt(psyshr.C, N; P = P*PC, nperiod = div(K,PC)),hr2bt(psyshr.D, N; P = P*PD, nperiod = div(K,PD))) 
    sys = dss(hr2btupd(psyshr.A, N; P, nperiod = K), hr2bt(psyshr.B, N; P, nperiod = K),
              hr2bt(psyshr.C, N; P, nperiod = K), hr2bt(psyshr.D, N; P, nperiod = K)) 
    return sys  
end      
"""
     hr2bt(Ahr::HarmonicArray, N; P, nperiod]) -> Abt::Matrix 

Build the block Toeplitz matrix of a harmonic (Fourier) array representation of a periodic matrix. 

The harmonic representation object `Ahr` of period `T` of a periodic matrix `Ahr(t)` 
of subperiod `T′ = T/k` is in the form

                     p
     Ahr(t) = A_0 +  ∑ ( Ac_i*cos(i*t*2*π/T′)+As_i*sin(i*2*π*t/T′) ) ,
                    i=1 

where `k ≥ 1` is the number of subperiods. `Ahr(t)` can be equivalently expressed in the Fourier series
representation form

                p
     Ahr(t) =   ∑ A_i*exp(im*i*ωh*t) ,
               i=-p

where `ωh = 2π/T′`, `A_i = (Ac_i-im*As_i)/2` and  `A_{-i} = (Ac_i+im*As_i)/2`. 
`N` is the number of selected harmonic components (or Fourier modes) used for approximation. 
The keyword parameter `P` is the number of full periods to be considered (default: `P = 1`) and `nperiod` is the
number of subperiods to be considered, such that `1 ≤ nperiod ≤ k` (default: `nperiod = k`). 

For a given number `N ≥ p`, if the number of period is `P = 1` and the number of subperiods is `nperiod = 1`, 
then the _banded_ block Toeplitz matrix `Abt` with `(2N+1)×(2N+1)` blocks is constructed
           
           ( A_0  A_{-1} …  A_{-p}        0    )
           ( A_1   A_0             ⋱           )
           (  ⋮         ⋱            ⋱         )
     Abt = ( A_p             ⋱          A_{-p} )
           (        ⋱           ⋱         ⋮    )
           (  0        A_p      …         A_0  )

If `N < p`, then a truncated _full_ block Toeplitz matrix is built using the first `N` harmonic components. 

Generally, for given `P ≥ 1` and  `nperiod ≥ 1`, the block Toeplitz matrix `Abt` is constructed with `(2N*np+1)×(2N*np+1)` blocks,
with `np = P*nperiod`, such that each `A_i` is preceeded in its column by `np-1` zero blocks, 
each `A_{-i}` is preceeded in its row by `np-1` zero blocks and all diagonal blocks are equal to`A_0`.   
"""
function hr2bt(A::HarmonicArray, N::Int; P::Int = 1, nperiod::Int = A.nperiod)
    N > 0 || error("the number of harmonic components must be nonnegative, got $N")
    #(nperiod < 1 || nperiod > A.nperiod) && error("number of subperiods must be between 1 and $(A.nperiod), got $nperiod")
    nperiod < 1  && error("number of subperiods must be between 1 and $(A.nperiod), got $nperiod")
    P < 1 && error("number of period must be at least 1, got $P")
    T = promote_type(Float64,eltype(A))
    p = size(A.values,3)-1
    p < 0 && (return zeros(complex(T),0,0))
    m, n = size(A)
    np = P*nperiod
    nb = 2*N*np+1
    BT = zeros(complex(T),nb*m,nb*n)
    i1, i2, j1, j2 = 1, m, 1, n
    BT[i1:i2,j1:j2] = A.values[:,:,1] # the diagonal A0
    ki1 = m*np+1
    ki2 = ki1+m-1
    kj1 = n*np+1
    kj2 = kj1+n-1
    minpN = min(p,N)
    for k = 1:minpN
        BT[i1:i2,kj1:kj2] = A.values[:,:,k+1]/2           # this is A_{-k} := (Ac_k+im*As_k)/2 
        BT[ki1:ki2,j1:j2] = conj(view(BT,i1:i2,kj1:kj2))  # this is A_k    := (Ac_k-im*As_k)/2 
        ki1 += m*np
        ki2 += m*np
        kj1 += n*np
        kj2 += n*np
    end
    i1 = 1
    j1 = n+1
    ik1 = m+1
    ik2 = 2m
    jk1 = n+1
    jk2 = 2n
    for i = 1:nb-1
        i1 += m
        i2 = min(i1+m*(minpN*np+1)-1,nb*m)
        i1 <= i2 || continue
        j1 += n
        j2 = min(j1+n*np-1,nb*n)
        BT[i1:i2,jk1:jk2] = BT[1:min(i2-i1+1,nb*m),1:n]
        BT[ik1:ik2,j1:j2] = BT[1:m,n+1:n+min(j2-j1+1,nb*n)]
        ik1 += m
        ik2 += m
        jk1 += n
        jk2 += n
    end
    return BT
end
"""
     hr2btupd(Ahr::HarmonicArray, N; P, nperiod, shift]) -> Ab::Matrix 

Build the updated block Toeplitz matrix of a harmonic (Fourier) array representation of a periodic matrix. 

If `BT` is the block Toeplitz matrix of the harmonic array representation of the `n × n` periodic matrix `Ahr` of subperiod `T′` 
(see [`HarmonicArray`](@ref)) as constructed with [`hr2bt`](@ref), then the updated matrix Ab = BT-NT is constructed, 
with `NT` a block-diagonal matrix with `n × n` diagonal blocks.
The `k`-th diagonal block of `NT` is the diagonal matrix `im*(μ + k*ωh)*I`, where `μ` is a shift specified via 
the keyword parameter `shift = μ` (default: `μ = 0`)  and `ωh` is the base frequency computed as `ωh = 2π*nperiod/(P*T′)`. 
The value of shift must satisfy `0 ≤ μ ≤ ωh/2`. 
"""
function hr2btupd(A::HarmonicArray, N::Int; P::Int = 1, nperiod::Int = A.nperiod, shift::Real = 0)
    n = size(A,1)
    n == size(A,2) || error("the periodic matrix must be square") 
    BT = hr2bt(A, N; P, nperiod)
    np = P*nperiod
    nb = 2*N*np+1
    ωh = 2*pi/P/A.period/A.nperiod*nperiod
    (shift >= 0 && shift <= ωh/2) 
    #Ej0 = fill(eltype(BT)(shift + im*ωh),n)
    Ej0 = similar(Vector{eltype(BT)},n)
    k = -N*np
    for i = 1:nb
        ind = (i-1)*n+1:i*n
        BT0 = view(BT,ind,ind)
        #BT0[diagind(BT0)] = diag(BT0)-k*Ej0;
        BT0[diagind(BT0)] = diag(BT0)-fill!(Ej0,(im*(shift + k*ωh)))
        k += 1
    end
    return BT
end

# function phasemat(A::HarmonicArray{:c,T}, N::Int; P::Int = 1, nperiod::Int = A.nperiod, shift::Real = 0) where T
#     (nperiod < 1 || nperiod > A.nperiod) && error("number of subperiods must be between 1 and $(A.period), got $nperiod")
#     P < 1 && error("number of period must be at least 1, got $P")
#     T1 = promote_type(Float64,T)
#     n, m = size(A)
#     n == m || error("the periodic matrix must be square") 
#     #nperiod = A.nperiod
#     np = P*nperiod
#     nb = 2*N*np+1
#     #ωh = 2*pi/A.period
#     ωh = 2*pi/A.period/nperiod
#     ωh = 2*pi/P/A.period/A.nperiod*nperiod
#     E = Diagonal((shift + im*ωh)*Matrix{complex(T)}(I,nb*n,nb*n))
#     k = -N*np
#     for i = 1:nb
#         ind = (i-1)*n+1:i*n
#         rmul!(view(E,ind,ind),k)
#         #k += nperiod
#         k += 1
#     end
#     return E
# end
# DiagDerOp(D::ApproxFunBase.DerivativeWrapper, n::Int) = 
#    eval(Meta.parse(string("[",join([string(join(["0I " for j in 1:i-1]),"D ",join(["0I " for j in i+1:n]),";") for i = 1:n]),"]")))
# DiagDerOp(n::Int) = 
#    string("[",join([string(join(["0I " for j in 1:i-1]),"D ",join(["0I " for j in i+1:n]),";") for i = 1:n]),"]")

function DiagDerOp(D::Union{ApproxFunBase.DerivativeWrapper,ApproxFunBase.ConstantTimesOperator}, n::Int) 
    Z = tuple(D,ntuple(n->0I,n-1)...)
    for i = 2:n
        Z1 = tuple(ntuple(n->0I,i-1)...,D,ntuple(n->0I,n-i)...)
        Z = tuple(Z...,Z1...)
    end
    return hvcat(n,Z...)
end

