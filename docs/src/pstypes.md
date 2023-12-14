# Constructors for periodic matrices

* **[`PeriodicMatrix`](@ref)**   Discrete-time periodic matrix representation.
* **[`PeriodicArray`](@ref)**    Discrete-time periodic array representation.
* **[`SwitchingPeriodicMatrix`](@ref)** Discrete-time switching periodic matrix representation.
* **[`SwitchingPeriodicArray`](@ref)** Discrete-time switching periodic array representation.
* **[`PeriodicFunctionMatrix`](@ref)**  Continuous-time periodic function matrix representation.
* **[`PeriodicSymbolicMatrix`](@ref)**   Continuous-time periodic symbolic matrix representation.
* **[`PeriodicTimeSeriesMatrix`](@ref)**   Continuous-time periodic time series matrix representation.
* **[`HarmonicArray`](@ref)**   Continuous-time harmonic array representation.
* **[`FourierFunctionMatrix`](@ref)**   Continuous-time Fourier functin matrix representation.
* **[`PeriodicSwitchingMatrix`](@ref)** Continuous-time switching periodic matrix representation.

```@docs
PeriodicMatrix
PeriodicArray
SwitchingPeriodicMatrix
SwitchingPeriodicArray
PeriodicFunctionMatrix
PeriodicSymbolicMatrix
PeriodicTimeSeriesMatrix
HarmonicArray
HarmonicArray(A0::MT, Acos::Union{Nothing, Vector{MT}}, Asin::Union{Nothing, Vector{MT}}, period::Real) where {T<:Real, MT<:VecOrMat{T}} 
FourierFunctionMatrix
PeriodicSwitchingMatrix
```
