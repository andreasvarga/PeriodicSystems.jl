# Release Notes


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
