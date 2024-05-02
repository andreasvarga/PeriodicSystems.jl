# Release Notes

## Version 0.7 

The following new functions have been implemented:
- `pclqr`, `pclqry`, `pdlqr`, `pdlqry` LQ-optimal state feedack based stabilization of periodic systems 
- `pckeg`, `pckegw`, `pdkeg`, `pdkegw` Kalman estimator gain matrix for periodic systems 
- `pcpofstab_sw`, `pcpofstab_hr` LQ-optimal output feedback stabilization of continuopus-time periodic systems
- `pdpofstab_sw`, `pdpofstab_hr` LQ-optimal output feedback stabilization of discrete-time periodic systems
- `pclqofc_sw`, `pclqofc_hr` LQ-optimal output feedback stabilization of continuopus-time periodic systems
- `pdlqofc`, `pdlqofc_sw` LQ-optimal output feedback stabilization of discrete-time periodic systems
- `pssfeedback` Periodic state feedback connection.
- `pssofeedback` Periodic state feedback with state estimator connection.

The following new supporting functions have been implemented:
- `pclyap2` to solve solve a pair of periodic continuous-time Lyapunov equations
- `pdlyap2` to solve solve a pair of periodic discrete-time Lyapunov equations
- `pslyapd2` to solve solve a pair of periodic Lyapunov equations using preallocated workspace 
- `pdlyaps2!` to solve solve a pair of reduced periodic Lyapunov equations using preallocated workspace 
- tools for efficient operations leading to symmetric periodic matrices  compute the symmetric matrix X = Y + transpose(Y) for a periodic matrix Y
- `pmmuladdsym` to efficiently compute the symmetrix periodic matrix X = α*A +  β*B*C
- `pmmultraddsym` to efficiently compute the symmetrix periodic matrix X = α*A +  β*transpose(B)*C

New versions of the following functions have been implemented: 
- `mb03vw!` and `mb03bd!` with explicit allocation of integer and real workspaces
- `pschur!` with a new interface to use preallocated storage for the transformation matrix
- `promote_period` with enhanced handling of constant periodic matrices
- various operations on periodic matrices with enhanced handling of constant periodic matrices 
- `pdlyap` and `pslyapd` to solve solve periodic Lyapunov equations with optional stability check 
- allocation free solvers of low order periodic Sylvester equations based on [`FastLapackInterface.jl`](https://github.com/DynareJulia/FastLapackInterface.jl)

The following extensions have been implemented:
- new periodic matrix type: switching periodic array 
- solution of periodic Lyapunov equations for discrete-time switching periodic arrays 
- enhanced version of function `psc2d` to determine discretized systems of arbitrary types

## Version 0.6.2 

New functions have been implemented to perform several connections of periodic systems (i.e., parallel, series, concatenations, appending, feedback) and the inversion operation. 
These functions are based on enhancements of basic concatenation functions on periodic matrices and a new set of functions which implement the block-diagonal stacking of periodic matrices.   

## Version 0.6.1 

The following new functions have been implemented:
- solution of periodic Lyapunov differential equations with nonnegative solutions for their square-root factors

The following breaking changes have been performed:
- new interfaces for the solvers of periodic Riccati matrix differential equations 

## Version 0.6 

This minor version includes several breaking changes, extensions and new functions. Also several bug fixes have been performed.

The following breaking changes have been performed:
- new definitions of norms and trace of periodic matrices  
- new interfaces for the solvers of periodic Lyapunov matrix differential equations 

The following extensions have been implemented:
- new periodic matrix types to support both continuous-time and discrete-time switching periodic matrices 
- new operations on periodic matrices, such as horizontal and vertical concatenations of periodic matrices
- solution of periodic Lyapunov equations for discrete-time switching periodic matrices 
- solution of periodic Lyapunov equations for continuous-time periodic switching matrices 

New functions have been implemented for the solution of periodic discrete-time Lyapunov equations with nonnegative solutions for their square-root factors. 

## Version 0.5.2

This patch version implements functions for the evaluation of time responses of continuous-time and discrete-time periodic systems and implementing `Base.getindex` for periodic matrices and systems.   
Functions are provided for the evaluation of the Hankel-norm of continuous-time and discrete-time periodic systems.   

## Version 0.5.1

This patch version implements functions for the evaluation of the H2-norm of continuous-time and discrete-time periodic systems. Several enhancements of Lyapunov differential equation solvers have been also performed.  

## Version 0.5 

This minor version concludes the development of periodic Riccati equation solvers, by implementing functions for the solution of periodic difference Riccati equations. Also several bug fixes have been performed.

## Version 0.4.2

This patch version implements functions for the solution of periodic differential Riccati equations. A new wrapper has been implemented for the SLICOT routine `MB03KD` to reorder the eigenvalues of a formal matrix product. This patch version also implements a collection of functions to compute real periodic Schur decompositions based on wrappers of SLICOT routines.   

## Version 0.4.1

This patch version implements additional operations for periodic matrices (e.g., inversion) and extends some existing methods to work with UniformScaling. Two functions for chopping and truncation of Harmonic Array representations have been implemented. 

## Version 0.4 

This minor version includes several new functions for the solution of continuous- and discrete-time periodic Lyapunov equations and implements basic operations with continuous- and discrete-time periodic matrices. 
An enhanced version of funtion `psc2d` is provided, which fully exploits the structure information of the underlying periodic matrices. 

## Version 0.3.1

This patch version relies on a new wrapper for the SLICOT subroutine MB03VW available in release v5.8 and includes several new functions for the computation of periodic Hessenberg and Schur decompositions. 

## Version 0.3 

This minor version includes new functions for the computation of poles and zeros of periodic systems. 

## Version 0.2

This minor release includes new functions for the computation of characteristic exponents of continuous-time periodic state matrices using frequency-lifting based on Harmonic Array and Fourier Function Matrix representations. 
For continuous-time periodic systems two functions have been implemented to build truncated 
Toeplitz operators based frequency lifted LTI systems (complex and real). 
For discrete-time periodic systems a function has been implemented to build several types of lifted LTI representations.


## Version 0.1.0

This is the initial release providing prototype implementations of several periodic array objects 
which served for the definition of the basic continuous-time and discrete-time periodic system objects. Several basic computational functions as the periodic Schur form, eigenvalues of matrix products, computation of monodromy matrix and computation of characteristic exponents/multipliers have been implemented. Two conversion functions of continuous-time periodic models (discretization, averaging) and a function for discretization/resampling of multirate LTI models have been implemented. 
