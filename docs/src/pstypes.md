# Constructors for periodic matrices

* **[`PeriodicMatrix`](@ref)**   Discrete-time periodic matrix representation.
* **[`PeriodicArray`](@ref)**    Discrete-time periodic array representation.
* **[`PeriodicFunctionMatrix`](@ref)**  Continuous-time periodic function matrix representation.
* **[`PeriodicSymbolicMatrix`](@ref)**   Continuous-time periodic symbolic matrix representation.
* **[`PeriodicTimeSeriesMatrix`](@ref)**   Continuous-time periodic time series matrix representation.
* **[`HarmonicArray`](@ref)**   Continuous-time harmonic array representation.
* **[`FourierFunctionMatrix`](@ref)**   Continuous-time Fourier functin matrix representation.

```@docs
PeriodicMatrix
PeriodicArray
PeriodicFunctionMatrix
PeriodicSymbolicMatrix
PeriodicTimeSeriesMatrix
HarmonicArray
HarmonicArray(A0::MT, Acos::Union{Nothing, Vector{MT}}, Asin::Union{Nothing, Vector{MT}}, period::Real) where {T<:Real, MT<:VecOrMat{T}} 
FourierFunctionMatrix
```
