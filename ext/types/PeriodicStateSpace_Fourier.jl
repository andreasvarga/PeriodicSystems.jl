function PeriodicSystems.PeriodicStateSpace(A::FFM1, B::FFM2, C::FFM3, D::FFM4) where {FFM1 <: FourierFunctionMatrix, FFM2 <: FourierFunctionMatrix, FFM3 <: FourierFunctionMatrix, FFM4 <: FourierFunctionMatrix}
    period = ps_validation(A, B, C, D)
    T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
    PeriodicStateSpace{FourierFunctionMatrix{:c,T,Fun}}((period == A.period && T == eltype(A)) ? A : FourierFunctionMatrix{:c,T}(A,period), 
                                                     (period == B.period && T == eltype(B)) ? B : FourierFunctionMatrix{:c,T}(B,period), 
                                                     (period == C.period && T == eltype(C)) ? C : FourierFunctionMatrix{:c,T}(C,period), 
                                                     (period == D.period && T == eltype(D)) ? D : FourierFunctionMatrix{:c,T}(D,period), 
                                                     Float64(period))
end
# function PeriodicStateSpace(A::FFM1, B::FFM2, C::FFM3, D::FFM4) where {FFM1 <: FourierFunctionMatrix, FFM2 <: FourierFunctionMatrix, FFM3 <: FourierFunctionMatrix, FFM4 <: FourierFunctionMatrix}
#     period = ps_validation(A, B, C, D)
#     T = promote_type(eltype(A),eltype(B),eltype(C),eltype(D))
#     PeriodicStateSpace{FourierFunctionMatrix{:c,T}}((period == A.period && T == eltype(A)) ? A : FourierFunctionMatrix{:c,T}(A,period), 
#                                                      (period == B.period && T == eltype(B)) ? B : FourierFunctionMatrix{:c,T}(B,period), 
#                                                      (period == C.period && T == eltype(C)) ? C : FourierFunctionMatrix{:c,T}(C,period), 
#                                                      (period == D.period && T == eltype(D)) ? D : FourierFunctionMatrix{:c,T}(D,period), 
#                                                      Float64(period))
# end


function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, sys::PeriodicStateSpace{<:FourierFunctionMatrix})
    summary(io, sys); println(io)
    n = size(sys.A,1) 
    p, m = size(sys.D)
    T = eltype(sys)
    if n > 0
       nperiod = sys.A.nperiod
       println(io, "\nState matrix A::$T($n×$n): subperiod: $(sys.A.period/nperiod)    #subperiods: $nperiod ")
       PeriodicMatrices.isconstant(sys.A) ? show(io, mime, sys.A.M(0)) : show(io, mime, sys.A.M)
       if m > 0 
          nperiod = sys.B.nperiod
          println(io, "\n\nInput matrix B::$T($n×$m): $(sys.B.period/nperiod)    #subperiods: $nperiod ") 
          PeriodicMatrices.isconstant(sys.B) ? show(io, mime, sys.B.M(0)) : show(io, mime, sys.B.M)
       else
          println(io, "\n\nEmpty input matrix B.")
       end
       
       if p > 0 
          nperiod = sys.C.nperiod
          println(io, "\n\nOutput matrix C::$T($p×$n): $(sys.C.period/nperiod)    #subperiods: $nperiod ")
          PeriodicMatrices.isconstant(sys.C) ? show(io, mime, sys.C.M(0)) : show(io, mime, sys.C.M)
       else 
          println(io, "\n\nEmpty output matrix C.") 
       end
       if m > 0 && p > 0
          nperiod = sys.D.nperiod
          println(io, "\n\nFeedthrough matrix D::$T($p×$m): $(sys.D.period/nperiod)    #subperiods: $nperiod ") 
          PeriodicMatrices.isconstant(sys.D) ? show(io, mime, sys.D.M(0)) : show(io, mime, sys.D.M)
       else
          println(io, "\n\nEmpty feedthrough matrix D.") 
       end
       println(io, "\n\nContinuous-time periodic state-space model.")  
    elseif m > 0 && p > 0
       nperiod = sys.D.nperiod
       println(io, "\nFeedthrough matrix D::$T($p×$m): $(sys.D.period/nperiod)    #subperiods: $nperiod ")
       PeriodicMatrices.isconstant(sys.D) ? show(io, mime, sys.D.M(0)) : show(io, mime, sys.D.M)
       println(io, "\n\nTime-varying gain.") 
    else
       println(io, "\nEmpty state-space model.")
    end
end

 

