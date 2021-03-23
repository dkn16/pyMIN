# pyMFs
A python code to calculate 3D Minkowski Functionals

Basically a rewrite of Liron Gleser's code, see https://arxiv.org/pdf/astro-ph/0602616

This code uses the Koenderink method to calculate.

Enviroment requirement: numpy, numba and sympy. For sympy you can just substitude the levicivita function with a ndarray and abandon the sympy.
