# marin
Mersenne Prime search program

## About

**marin** is an [OpenCL™](https://www.khronos.org/opencl/) application.  
It determines whether a [Mersenne number](https://en.wikipedia.org/wiki/Mersenne_prime) 2<sup>*p*</sup> - 1 is a probable prime using [Fermat primality test](https://en.wikipedia.org/wiki/Fermat_primality_test) (*a* = 3) or prime using [Lucas-Lehmer primality test](https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test).  

It implements an [Efficient Modular Exponentiation Proof Scheme](https://arxiv.org/abs/2209.15623). The prp test is validated with [Gerbicz - Li](https://www.mersenneforum.org/showthread.php?t=22510) error checking.  

Efficient multiplication modulo a Mersenne number is evaluated using an [Irrational Base Discrete Weighted Transform](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/).  
A Number Theoretic Transform is implemented, over the field Z/*p*Z, *p* = 2<sup>64</sup>&nbsp;-&nbsp;2<sup>32</sup>&nbsp;+&nbsp;1 (see: Nick Craig-Wood, [IOCCC 2012 Entry](https://github.com/ncw/ioccc2012/)).  

The algorithm is different from Nick Craig-Wood's implementation:
 - By the use of weights, *x*<sup>*p*</sup> - 1 is transformed into *x*<sup>*n*</sup> - 1, where *n* = 2<sup>*m*</sup> or 5&nbsp;&middot;&nbsp;2<sup>*m*</sup>.
 - A recursive polynomial factorization approach splits *x*<sup>2*n*</sup> - *r*<sup>2</sup> into *x*<sup>*n*</sup> - *r*, *x*<sup>*n*</sup> + *r* and *x*<sup>5*n*</sup> - *r*<sup>5</sup> into 5 polynomials of the form *x*<sup>*n*</sup> - *r*<sub>*i*</sub>. See: [Fast Multiplication](https://github.com/galloty/FastMultiplication).
 - Butterfly sizes are radix-4 and radix-5.  

[*algo*](algo/) contains basic implementations of the algorithm. Two main aplications are generated:
- *marin_cpu* is an implementation of the algorithm on CPU. It is not optimised, it helps to check and debug the OpenCL application.
- *marin* is the optimised OpenCL application. It can test any prime exponent *p* in [7; 1,509,949,421].  

A checkpoint file is created if the application is interrupted and *marin* resumes from a checkpoint if it exists.

## Build

The compiler must support the built-in __uint128_t type (GCC and Clang) or the _umul128 instruction (Visual Studio).  
The code has been validated with GCC 15.2 on Windows ([MSYS2](https://www.msys2.org/)) and Linux.  
