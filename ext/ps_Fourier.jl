# function PeriodicSystems.ps(PMT::Type, sys::DST, period::Real; ns::Int = 1) where {PMT <: FourierFunctionMatrix,  DST <: DescriptorStateSpace}
#     sys.E == I || error("only standard state-spece models supported")
#     Ts = sys.Ts
#     Ts == 0 || error("only continuous periodic matrix types allowed")
#     ps(PMT(sys.A,period), PMT(sys.B,period), PMT(sys.C,period), PMT(sys.D,period))
# end
# function ps(D::PM) where {PM <: FourierFunctionMatrix}
#     error("This function is not available for FourierFunctionMatrix type")
#     # p, m = size(D,1), size(D,2)
#     # ps(FourierFunctionMatrix(zeros(T,0,0),D.period),
#     # FourierFunctionMatrix(zeros(T,0,m),D.period), 
#     # FourierFunctionMatrix(zeros(T,p,0),D.period), D)
# end
