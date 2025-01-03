# function PeriodicSystems.ps(PMT::Type, sys::DST, period::Real; ns::Int = 1) where {PMT <: FourierFunctionMatrix,  DST <: DescriptorStateSpace}
#     sys.E == I || error("only standard state-spece models supported")
#     Ts = sys.Ts
#     Ts == 0 || error("only continuous periodic matrix types allowed")
#     ps(PMT(sys.A,period), PMT(sys.B,period), PMT(sys.C,period), PMT(sys.D,period))
# end
