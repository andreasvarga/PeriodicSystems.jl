module Runtests

using Test, PeriodicSystems

@testset "Test PeriodicSystems" begin
# test constructors, basic tools
include("test_psutils.jl")
include("test_psconnect.jl")
include("test_psconversions.jl")
include("test_pslifting.jl")
include("test_psanalysis.jl")
include("test_pstimeresp.jl")
include("test_stabilization.jl")
end

end
