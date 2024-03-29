# pyMIN

## Introduction
A python code to calculate 3D Minkowski Functionals

Basically a rewrite of Liron Gleser's code, see https://arxiv.org/pdf/astro-ph/0602616

This code uses the Koenderink method to calculate the MFs.

Enviroment requirement: numpy, numba and sympy. For sympy you can just substitude the levicivita function with a ndarray and abandon the sympy.

Function `calculateMFs` Calculate the 3D MFs ($V_{0}-V_{3}$) of a given field (must be 3D).

## Example
Please download (only) pyMIN.py to one of the system paths. Or, you can download it to any folder and add that folder to system paths:
```
import sys
sys.path.append('~/where/you/download/the/script/')
```
Then you can import and calculate.
```
import pyMIN as pm
import numpy as np

data = np.random.normal((64,64,64))
v0,v1,v2,v3 = calculateMFs(data)
```

