module Runtests

using Test, PeriodicSystems

@testset "Test PeriodicSystems" begin
# test constructors, basic tools
include("test_pschur.jl")
# include("test_psutils.jl")
# include("test_pmops.jl")
# include("test_conversions.jl")
# include("test_pslifting.jl")
# include("test_psanalysis.jl")
# include("test_pslyap.jl")
# include("test_psclyap.jl")
# include("test_pscric.jl")
end

end
