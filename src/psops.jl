# sum psys1+psys2
function +(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   size(psys1) == size(psys2) || error("The systems maps have different shapes.")
   A = PeriodicMatrices.blockdiag(psys1.A,psys2.A)
   B = [psys1.B ; psys2.B]
   C = [psys1.C psys2.C]
   D = psys1.D + psys2.D
   return ps(A, B, C, D)
end
"""
    psys = psparallel(psys1, psys2)
    psys = psys1 + psys2

Build the parallel connection `psys` of periodic systems `psys1` and `psys2`. This coupling formally corresponds to the addition of their transfer maps as `psys = psys1 + psys2`.
"""
function psparallel(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   return psys1+psys2
end
# difference psys1-psys2
function -(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   size(psys1) == size(psys2) || error("The systems maps have different shapes.")
   A = PeriodicMatrices.blockdiag(psys1.A,psys2.A)
   B = [psys1.B; psys2.B]
   C = [psys1.C -psys2.C]
   D = psys1.D - psys2.D
   return ps(A, B, C, D)
end

# negation -psys
function -(psys::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   return ps(psys.A, psys.B, -psys.C, -psys.D)
end
# product psys2*psys1
function *(psys1::PeriodicStateSpace{PM1}, psys2::PeriodicStateSpace{PM2}) where {T1, T2, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}}
   size(psys1.D,2) == size(psys2.D,1) || error("psys1 must have same number of inputs as psys2 has outputs")
   n1 = size(psys1.A,1)
   n2 = size(psys2.A,1)

   A = [[psys1.A  psys1.B*psys2.C];
        [zeros(eltype(psys2.A),n2,n1) psys2.A]]
   B = [psys1.B*psys2.D ; psys2.B]
   C = [psys1.C psys1.D*psys2.C]
   D = psys1.D*psys2.D
   return ps(A, B, C, D)
end
function *(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: PeriodicMatrix}
   size(psys1.D,2) == size(psys2.D,1) || error("psys1 must have same number of inputs as psys2 has outputs")
   n1 = size(psys1.A,2)
   n2 = size(psys2.A,1)
   T = eltype(psys2.A)
   A = [[psys1.A  psys1.B*psys2.C];
        [PeriodicMatrix{:d,T}(PeriodicMatrices.pmzeros(T,n2,n1),psys2.A.period) psys2.A]] 
   B = [psys1.B*psys2.D ; psys2.B]
   C = [psys1.C psys1.D*psys2.C]
   D = psys1.D*psys2.D
   return ps(A, B, C, D)
end
function *(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: PeriodicArray}
   size(psys1.D,2) == size(psys2.D,1) || error("psys1 must have same number of inputs as psys2 has outputs")
   n1 = size(psys1.A,2)
   n2 = size(psys2.A,1)
   T = eltype(psys2.A)
   A = [[psys1.A  psys1.B*psys2.C];
        [PeriodicArray{:d,T}(zeros(T,n2,n1,1),psys2.A.period; nperiod = round(Int,psys2.A.period/psys2.Ts)) psys2.A]] 
   B = [psys1.B*psys2.D ; psys2.B]
   C = [psys1.C psys1.D*psys2.C]
   D = psys1.D*psys2.D
   return ps(A, B, C, D)
end
function *(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: SwitchingPeriodicMatrix}
   size(psys1.D,2) == size(psys2.D,1) || error("psys1 must have same number of inputs as psys2 has outputs")
   n1 = size(psys1.A,2)
   n2 = size(psys2.A,1)
   T = eltype(psys2.A)
   A = [[psys1.A  psys1.B*psys2.C];
        [convert(SwitchingPeriodicMatrix,PeriodicMatrix{:d,T}(PeriodicMatrices.pmzeros(T,n2,n1),psys2.A.period)) psys2.A]] 
   B = [psys1.B*psys2.D ; psys2.B]
   C = [psys1.C psys1.D*psys2.C]
   D = psys1.D*psys2.D
   return ps(A, B, C, D)
end


# sI*psys
function *(s::Union{UniformScaling{T},T}, psys::PeriodicStateSpace{PM}) where {T <: Real, PM <: AbstractPeriodicArray}
   return ps(psys.A, psys.B, s*psys.C, s*psys.D)
end
# psys*sI
function *(psys::PeriodicStateSpace{PM},s::Union{UniformScaling{T},T}) where {T <: Real, PM <: AbstractPeriodicArray}
   return ps(psys.A, psys.B*s, psys.C, psys.D*s)
end
"""
    psys = psseries(psys1, psys2)
    psys = psys2*psys1

Build the series connection `psys` of periodic systems `psys1` and `psys2`. This coupling formally corresponds to the product of their transfer maps as `psys = psys2*psys1`.
"""
function psseries(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   return psys2*psys1
end
"""
    psys = psappend(psys1, psys2) 

Append the periodic systems `psys1` and `psys2` by concatenating their input and output vectors. 
This corresponds to build `psys` as the block diagonal concatenation of their transfer maps. 
"""
function psappend(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   A = PeriodicMatrices.blockdiag(psys1.A, psys2.A)
   B = PeriodicMatrices.blockdiag(psys1.B, psys2.B)
   C = PeriodicMatrices.blockdiag(psys1.C, psys2.C)
   D = PeriodicMatrices.blockdiag(psys1.D, psys2.D)
   return ps(A, B, C, D)
end

"""
    psys = pshorzcat(psys1,psys2)
    psys = [psys1 psys2]

Concatenate horizontally the two periodic systems `psys1` and `psys2` 
by concatenating their input vectors. This formally corresponds to the horizontal 
concatenation of their transfer maps. 
"""
function pshorzcat(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   return [psys1 psys2]
end
function hcat(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   A = PeriodicMatrices.blockdiag(psys1.A, psys2.A)
   B = PeriodicMatrices.blockdiag(psys1.B, psys2.B)
   C = [psys1.C psys2.C]
   D = [psys1.D psys2.D]
   return ps(A, B, C, D)
end
"""
    psys = psvertcat(psys1,psys2)
    psys = [psys1; psys2]

Concatenate vertically the two periodic systems `psys1` and `psys2` 
by concatenating their output vectors. This formally corresponds to the vertical 
concatenation of their transfer maps. 
"""
function psvertcat(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   return [psys1; psys2]
end
function vcat(psys1::PeriodicStateSpace{PM}, psys2::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   A = PeriodicMatrices.blockdiag(psys1.A, psys2.A)
   B = [psys1.B; psys2.B]
   C = PeriodicMatrices.blockdiag(psys1.C, psys2.C)
   D = [psys1.D; psys2.D]
   return ps(A, B, C, D)
end
"""
    psysi = psinv(psys)

Compute the inverse `psysi` of the square periodic system `psys`.  
This operation formally corresponds to the inversion of the transfer map of `psys` such that `psysi*psys` is the identity mapping.
"""
function psinv(psys::PeriodicStateSpace{PM}) where {PM <: AbstractPeriodicArray}
   D = inv(psys.D)
   C = -D*psys.C
   A = psys.A+psys.B*C
   B = psys.B*D
   return ps(A, B, C, D)
end


# Basic Operations
function ==(sys1::PeriodicStateSpace{PM1}, sys2::PeriodicStateSpace{PM2}) where {PM1 <: AbstractPeriodicArray,PM2 <: AbstractPeriodicArray}
   # fieldnames(DST1) == fieldnames(DST2) || (return false)
   return all(getfield(sys1, f) == getfield(sys2, f) for f in fieldnames(PeriodicStateSpace))
end

function isapprox(sys1::PeriodicStateSpace{PM1}, sys2::PeriodicStateSpace{PM2}; atol = zero(real(eltype(sys1))), 
                 rtol::Real =  (maximum(maximum.(size(sys1.A)))+1)*eps(real(float(one(real(eltype(sys1))))))*iszero(atol)) where 
                 {PM1 <: AbstractPeriodicArray,PM2 <: AbstractPeriodicArray}
   #fieldnames(DST1) == fieldnames(DST2) || (return false)
   return all(isapprox(getfield(sys1, f), getfield(sys2, f); atol = atol, rtol = rtol) for f in fieldnames(PeriodicStateSpace))
end


"""
     psyscl = psfeedback(psys, K, (inp, out); negative = false)

Build for a given periodic system `psys` with input vector `u` and output vector `y` and 
a periodic output feedback gain `K(t)` the closed-loop periodic system `psyscl`
corresponding to the memoryless output feedback `u[inp] = K(t)*y[out] + v`, where `inp` and `out` 
are indices, vectors of indices, index ranges, `:` or any combinations of them. Only distinct indices 
can be specified. If `negative = true`, a negative psfeedback `u[inp] = -K(t)*y[out] + v` is used.
"""
function psfeedback(psys::PeriodicStateSpace{PM}, K::PM1, inds = (Colon(),Colon()); negative::Bool = false) where {Domain,T, T1, PM <: AbstractPeriodicArray{Domain,T}, PM1 <: AbstractPeriodicArray{Domain,T1}}
   K1 = PM == PM1 ? K : convert(PM,K)
   size(inds, 1) != 2 &&
         error("Must specify 2 indices to index periodic state-space models")
   u1, y1 = index2range(inds...) 
   p, m = size(psys)
   u1 == Colon() && (u1 = 1:m) 
   y1 == Colon() && (y1 = 1:p)
 
   maximum(u1) > m && error("input indices must not exceed $m")
   maximum(y1) > p && error("output indices must not exceed $p")
   allunique(u1) || error("all input indices must be distinct")
   allunique(y1) || error("all output indices must be distinct")
   m1, p1 = maximum.(size(K))
   length(u1) == m1  || error("number of row indices must be equal to the number of rows of K")
   length(y1) == p1  || error("number of column indices must be equal to the number of columns of K")
   y2 = setdiff(Vector(1:p),Vector(y1))
   u2 = setdiff(Vector(1:m),Vector(u1))
   
   B1 = psys.B[:,u1]
   B2 = psys.B[:,u2]
   C1 = psys.C[y1,:]
   C2 = psys.C[y2,:]
   D11 = psys.D[y1,u1]
   D12 = psys.D[y1,u2]
   D21 = psys.D[y2,u1]
   D22 = psys.D[y2,u2]
   nullD11 = iszero(D11)
   Li = nullD11 ? I : (negative ? inv(I+K1*D11) : inv(I-K1*D11))
   KT = nullD11 ? (negative ? -K1 : K1) : (negative ? -Li*K1 : Li*K1)
   ip = sortperm([y1;y2])
   jp = sortperm([u1;u2])
   return ps(psys.A+B1*KT*C1, [B1*Li B2+B1*KT*D12][:,jp], [C1 + D11*KT*C1; C2 + D21*KT*C1][ip,:], 
                         [[D11*Li D12+D11*KT*D12]; [D21*Li D22+D21*KT*D12]][ip,jp])

end
# function psfeedback(psys::PeriodicStateSpace{PM}, K::PM1; negative::Bool = true) where {Domain,T, T1, PM <: AbstractPeriodicArray{Domain,T}, PM1 <: AbstractPeriodicArray{Domain,T1}}
#     psfeedback(psys,K,(1:maximum(size(K,2)),1:maximum(size(K,1))); negative)
# end
"""
     psyscl = psfeedback(sys, K, (inp, out); negative = false)

Build for a given standard  state-space system `sys` with input vector `u` and output vector `y` and 
a periodic output feedback gain `K(t)` the closed-loop periodic system `psyscl`
corresponding to the memoryless output feedback `u[inp] = K(t)*y[out] + v`, where `inp` and `out` are 
are indices, vectors of indices, index ranges, `:` or any combinations of them. Only distinct indices 
can be specified. If `negative = true`, a negative feedback `u[inp] = -K(t)*y[out] + v` is used.
For a continuous-time system `sys`, `K` must be a periodic switching matrix or a discrete-time periodic matrix, 
while for a discrete-time system `sys`, `K` must be a discrete-time periodic matrix with the same sample time. 
"""
function psfeedback(sys::DS, K::PM, inds = (Colon(),Colon()); negative::Bool = false) where {DS <: DescriptorStateSpace, PM <: PeriodicSwitchingMatrix}
    sys.E == I || error("descriptor systems with E ≠ I not supported")
    if sys.Ts == 0
       psys = ps(PeriodicSwitchingMatrix, sys.A, sys.B, sys.C, sys.D, K.period) 
       return psfeedback(psys,K,inds; negative)
    else
       Δ = abs(sys.Ts)
       ns = Int(floor(round(K.period/Δ)))
       psys = ps(sys,K.period; ns) 
       Kd = PeriodicMatrix([tpmeval(K,(i-1)*Δ) for i in 1:ns],K.period)
       return psfeedback(psys,Kd,inds; negative)
    end
end
function psfeedback(sys::DS, K::PM, inds = (Colon(),Colon()); negative::Bool = false) where {DS <: DescriptorStateSpace, PM <: PeriodicFunctionMatrix}
   sys.E == I || error("descriptor systems with E ≠ I not supported")
   sys.Ts == 0 || error("only coontinuous-time system is allowed")
   psys = ps(PeriodicFunctionMatrix, sys.A, sys.B, sys.C, sys.D, K.period) 
   return psfeedback(psys,K,inds; negative)
end
function psfeedback(sys::DS, K::PM, inds = (Colon(),Colon()); negative::Bool = false) where {DS <: DescriptorStateSpace, PM <: HarmonicArray}
   sys.E == I || error("descriptor systems with E ≠ I not supported")
   sys.Ts == 0 || error("only coontinuous-time system is allowed")
   psys = ps(HarmonicArray, sys.A, sys.B, sys.C, sys.D, K.period) 
   return psfeedback(psys,K,inds; negative)
end

function psfeedback(sys::DS, K::PM, inds = (Colon(),Colon()); negative::Bool = false) where {DS <: DescriptorStateSpace, PM <: PeriodicMatrix}
   sys.E == I || error("descriptor systems with E ≠ I not supported")
   if sys.Ts == 0
      Δ = K.Ts
      ns = Int(floor(round(K.period/Δ)))
      psys = ps(c2d(sys,Δ)[1],K.period; ns) 
      return psfeedback(psys,K,inds; negative)
   else
      ns = Int(floor(round(K.period/abs(sys.Ts))))
      abs(sys.Ts) == K.Ts || throw(ArgumentError("sys and K must have the same sample time"))
      psys = ps(sys,K.period; ns) 
      return psfeedback(psys,K,inds; negative)
   end
end
function psfeedback(sys::DS, K::PM, inds = (Colon(),Colon()); negative::Bool = false) where {DS <: DescriptorStateSpace, PM <: SwitchingPeriodicMatrix}
   sys.E == I || error("descriptor systems with E ≠ I not supported")
   if sys.Ts == 0
      
      
      etzΔ = K.Ts45
      ns = Int(floor(round(K.period/Δ)))
      #psys = ps(c2d(sys,Δ)[1],K.period; ns) 
      psys = convert(PeriodicStateSpace{SwitchingPeriodicMatrix}, ps(c2d(sys,Δ)[1],K.period; ns) ) 
      return psfeedback(psys,K,inds; negative)
   else
      ns = Int(floor(round(K.period/abs(sys.Ts))))
      abs(sys.Ts) == K.Ts || throw(ArgumentError("sys and K must have the same sample time"))
      #psys = convert(PeriodicStateSpace{SwitchingPeriodicMatrix}, ps(sys,K.period; ns)) 
      psys = ps(SwitchingPeriodicMatrix, sys, K.period) 
      return psfeedback(psys,K,inds; negative)
   end
end
"""
     psyscl = pssfeedback(psys, F, inp; negative = false)

Build for a given periodic system `psys` with input vector `u` and output vector `y` and 
a periodic state feedback gain `F(t)` the closed-loop periodic system `psyscl`
corresponding to the memoryless state feedback `u[inp] = F(t)*x + v`, where `inp`  
are indices, a vector of indices, an index range, `:` or any combinations of them. Only distinct indices 
can be specified. If `negative = true`, a negative state feedback `u[inp] = -F(t)*x + v` is used.
"""
function pssfeedback(psys::PeriodicStateSpace{PM}, F::PM1, inds = Colon(); negative::Bool = false) where {Domain,T, T1, PM <: AbstractPeriodicArray{Domain,T}, PM1 <: AbstractPeriodicArray{Domain,T1}}
   F1 = PM == PM1 ? F : convert(PM,F)
   u1 = index2range(inds) 
   m = size(psys,2)
   u1 == Colon() && (u1 = 1:m) 
 
   maximum(u1) > m && error("input indices must not exceed $m")
   allunique(u1) || error("all input indices must be distinct")
   m1 = maximum(size(F1,1))
   length(u1) == m1  || error("number of row indices must be equal to the number of rows of F")
   
   FT = negative ? -F1 : F1
   return ps(psys.A+psys.B[:,u1]*FT, psys.B, psys.C + psys.D[:,u1]*FT, psys.D)
end
"""
     psyscl = pssofeedback(psys, F, K, (inp, out); negative = false)

Build for a given periodic system `psys`, with input vector `u` and output vector `y`, 
a periodic state feedback gain `F(t)` and a periodic Kalman gain `K(t)` the closed-loop periodic system `psyscl`
corresponding to the memoryless state feedback `u[inp] = F(t)*xe + v` and 
a full state estimator with state `xe` and inputs `[u[inp]; y[out]]`, where `inp` and `out` 
are indices, vectors of indices, index ranges, `:` or any combinations of them. Only distinct indices 
can be specified. If `negative = true`, a negative state feedback `u[inp] = -F(t)*xe + v` is used.
"""
function pssofeedback(psys::PeriodicStateSpace{PM}, F::PM1, K::PM2, inds = (Colon(),Colon()); negative::Bool = false) where {Domain,T, T1, T2, PM <: AbstractPeriodicArray{Domain,T}, PM1 <: AbstractPeriodicArray{Domain,T1}, PM2 <: AbstractPeriodicArray{Domain,T2}}
   F1 = PM == PM1 ? F : convert(PM,F)
   K1 = PM == PM2 ? K : convert(PM,K)

   u1, y1 = index2range(inds...) 
   p, m = size(psys)
   u1 == Colon() && (u1 = 1:m) 
   y1 == Colon() && (y1 = 1:p)

   maximum(u1) > m && error("input indices must not exceed $m")
   maximum(y1) > p && error("output indices must not exceed $p")
   allunique(u1) || error("all input indices must be distinct")
   allunique(y1) || error("all output indices must be distinct")
   m1, n1 = maximum.(size(F))
   p1 = maximum(size(K,2))
   length(u1) == m1  || error("number of row indices must be equal to the number of rows of F")
   length(y1) == p1  || error("number of column indices must be equal to the number of columns of K")
   y2 = setdiff(Vector(1:p),Vector(y1))
   u2 = setdiff(Vector(1:m),Vector(u1))
   ip = sortperm([y1;y2])
   jp = sortperm([u1;u2])

   B1 = psys.B[:,u1]
   B2 = psys.B[:,u2]
   C1 = psys.C[y1,:]
   C2 = psys.C[y2,:]
   D1 = psys.D[:,u1]
   D11 = psys.D[y1,u1]
   D12 = psys.D[y1,u2]
   D21 = psys.D[y2,u1]
   D22 = psys.D[y2,u2]

   #FT = negative ? -F1 : F1
   (BF,DF) = negative ? (-B1*F1,-D1*F1) : (B1*F1,D1*F1)
   A = blockut(psys.A+BF, -BF, psys.A-K1*C1)
   B = blockut(B1,B2,B2-K1*D12)[:,jp]

   #return ps(A, [psys.B; zeros(T,n1,m)], [psys.C+DF -DF], psys.D)
   return ps(A, B, [psys.C+DF -DF], psys.D)
end


