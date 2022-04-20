module Runtests

using Test, PeriodicSystems

@testset "Test PeriodicSystems" begin
# test constructors, basic tools
include("test_psutils.jl")
include("test_conversions.jl")
end

end
