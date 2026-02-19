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

## Citation
Please also consider cite [Diao et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad6c40/meta)
```
@ARTICLE{2024ApJ...974..141D,
       author = {{Diao}, Kangning and {Chen}, Zhaoting and {Chen}, Xuelei and {Mao}, Yi},
        title = "{Reionization Parameter Inference from 3D Minkowski Functionals of the 21 cm Signals}",
      journal = {ApJ},
     keywords = {Cosmology, Reionization, H I line emission, Markov chain Monte Carlo, 343, 1383, 690, 1889, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = oct,
       volume = {974},
       number = {1},
          eid = {141},
        pages = {141},
          doi = {10.3847/1538-4357/ad6c40},
archivePrefix = {arXiv},
       eprint = {2406.20058},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...974..141D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
