#!/usr/bin/env python

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Take first derivative of a function
# ----------------------------------------------------------------------

def f(x):
    return(-2*x)

def explicit_euler(T, setp_size, init_cond):

    """ 
     Use Euler's method with different stepsizes to solve the IVP:
     dx/dt = -2*x, with x(0) = 3 over the time-horizon [0,2]

     Compare the numerical approximation of the IVP solution to its analytical
     solution by plotting both solutions in the same figure. 

    """
    xs = np.zeros(len(T))
    ts = np.zeros(len(T))
    xs[0] = init_cond
    ts[0] = 0
    for i in range(len(T)-1):
        xs[i+1] = xs[i] + setp_size*f(xs[i])
        ts[i+1] = ts[i] + setp_size

    return xs, ts

# ----------------------------------------------------------------------
# Take second derivative of a function
# ----------------------------------------------------------------------


def analytic(x):

    """ Function that gets analytic solutions

    Parameters
    ----------
    x - the location of the point at which f(x) is evaluated

    Notes
    -----
    These are analytic solutions!

    """

    f = -2*x
    cond = 3
    exact = 3*np.exp(-2*x)

    return f, cond, exact #, fctstring, dxstring, dx2string


if __name__ == "__main__":

    
    # arange doesn't include last point, so add explicitely:
    sz = 0.2
    T = np.arange(0,2,sz)    

    fn, cond, exact = analytic(T)
    xs, ts = explicit_euler(T, sz, 3)
    
    plt.plot(T,exact)
    plt.plot(ts,xs)    
    
    
    
    
    