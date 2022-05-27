```@meta
CurrentModule = PeriodicSystems
DocTestSetup = quote
    using PeriodicSystems
end
```

# PeriodicSystems.jl

[![DocBuild](https://github.com/andreasvarga/PeriodicSystems.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/PeriodicSystems.jl/actions)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/PeriodicSystems.jl)

`PeriodicSystems.jl` is intended to be a collection of Julia functions for numerical computations related to periodic system representations in the continuous-time form

     dx(t)/dt = A(t)x(t) + B(t)u(t) ,
     y(t)     = C(t)x(t) + D(t)u(t) ,

or in the discrete-time form

     x(t+1)  = A(t)x(t) + B(t)u(t) ,
     y(t)    = C(t)x(t) + D(t)u(t) ,

where `x(t)`, `u(t)` and `y(t)` are the system state vector, system input vector and system output vector, respectively, and `t` is the continuous or discrete time variable. The system matrices satisfy `A(t) = A(t+T)`, `B(t) = B(t+T)`, `C(t) = C(t+T)`, `D(t) = D(t+T)`,  i.e., are periodic with period `T`. 

Many control applications are formulated as genuine
periodic control problems as for example, satellite attitude control, helicopter forward flight control, orbital stabilization of underactuated systems, etc. Besides
that, periodic systems represent a general framework to analyze and design multi-rate sampled-data systems. 

The targeted functionality of this package is described in [1] and will cover both continuous-time and discrete-time periodic systems. The following categories of functions are implemented of planned:

**Building periodic system state-space models**

* **[`PeriodicStateSpace`](@ref)**   Periodic state space system representation.
* **[`ps`](@ref)**  Construction of periodic state-space models.
* **[`psmrc2d`](@ref)**  Discretization/resampling of a multirate LTI system. 

**Basic operations on periodic system models**

**Discretization of continuous-time periodic systems**

* **[`psc2d`](@ref)**  Discretization of a continuous-time periodic system. 

# Conversions to LTI representations

* **[`psaverage`](@ref)**  Computation of the averaged LTI system of a continuous-time periodic system.
* **[`psteval`](@ref)**  Computation of the time evaluated LTI system of a continuous-time periodic system.
* **[`ps2fls`](@ref)**  Computation of a complex frequency-lifted LTI representation of a continuous-time periodic system. 
* **[`ps2frls`](@ref)**  Computation of a real frequency-lifted LTI representation of a continuous-time periodic system. 
* **[`ps2ls`](@ref)**  Computation of a time-lifted LTI representation of a discrete-time periodic system. 

**Periodic system analysis**

* **[`pspole`](@ref)**  Computation of the poles of a periodic system.
* **[`pszero`](@ref)**  Computation of the zeros of a periodic system.
* **[`isstable`](@ref)**  Assessment of stability of a periodic system.

**Solving periodic matrix equations**

**Simplification of periodic system models**

**Periodic output and state feedback controller design** 



## [Release Notes](https://github.com/andreasvarga/PeriodicSystems.jl/blob/master/ReleaseNotes.md)

## Main developer

[Andreas Varga](https://sites.google.com/view/andreasvarga/home)

License: MIT (expat)

## References

[1] A. Varga. [A Periodic Systems Toolbox for Matlab](https://elib.dlr.de/12283/1/varga_ifac2005p1.pdf). Proc. of IFAC 2005 World Congress, Prague, Czech Republic, 2005.

[2] S. Bittanti and P. Colaneri. Periodic Systems - Filtering and Control, Springer Verlag, 2009.

[3]  A. Varga, [MatrixPencils.jl: Matrix pencil manipulation using Julia](https://github.com/andreasvarga/MatrixPencils.jl).
[Zenodo: https://doi.org/10.5281/zenodo.3894503](https://doi.org/10.5281/zenodo.3894503).

[4]  A. Varga, [MatrixEquations.jl: Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia](https://github.com/andreasvarga/MatrixEquations.jl). [Zenodo: https://doi.org/10.5281/zenodo.3556867](https://doi.org/10.5281/zenodo.3556867).
