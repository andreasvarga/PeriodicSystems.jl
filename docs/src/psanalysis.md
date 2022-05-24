# Analysis of periodic systems

* **[`pspole`](@ref)**  Computation of the poles of a periodic system.
* **[`pszero`](@ref)**  Computation of the zeros of a periodic system.

```@docs
pspole
pszero(psys::PeriodicStateSpace{Union{PeriodicFunctionMatrix, PeriodicSymbolicMatrix, PeriodicTimeSeriesMatrix}})
pszero(psys::PeriodicStateSpace{HarmonicArray})
pszero(psys::PeriodicStateSpace{FourierFunctionMatrix})
```
