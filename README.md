# PeriodicSystems.jl

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4568159.svg)](https://doi.org/10.5281/zenodo.4568159) -->
[![DocBuild](https://github.com/andreasvarga/PeriodicSystems.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/PeriodicSystems.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/PeriodicSystems.jl/coverage.svg?branch=main)](https://codecov.io/gh/andreasvarga/PeriodicSystems.jl?branch=main)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/PeriodicSystems.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/PeriodicSystems.jl/blob/main/LICENSE.md)

## Linear periodic time-varying systems 

## Compatibility

Julia 1.7 and higher.

<!-- ## How to install

````JULIA
pkg> add PeriodicSystems
pkg> test PeriodicSystems
```` -->

## About

`PeriodicSystems.jl` is intended to be a collection of Julia functions for numerical computations related to periodic system representations in the continuous-time form

     dx(t)/dt = A(t)x(t) + B(t)u(t) ,
     y(t)     = Cx(t) + Du(t) ,

or in the discrete-time form

     x(t+1)  = A(t)x(t) + B(t)u(t) ,
     y(t)(t) = C(t)x(t) + D(t)u(t) ,

where `x(t)`, `u(t)` and `y(t)` are the system state vector, system input vector and system output vector, respectively, and `t` is the continuous or discrete time variable. The system matrices satisfy `A(t) = A(t+T)`, `B(t) = B(t+T)`, `C(t) = C(t+T)`, `D(t) = D(t+T)`,  i.e., are periodic with period `T`. 

The targeted functionality of this package is described in [1]

[1] A. Varga. [A Periodic Systems Toolbox for Matlab](https://www.sciencedirect.com/science/article/pii/S1474667016364874). 
Proc. of IFAC 2005 World Congress, Prague, Czech Republic, 2005.

