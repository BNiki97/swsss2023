#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

# smoother equation - this is what conduction does
# a weighted average between the two neighboring values


if __name__ == "__main__":

    dx = 0.25

    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10 + 2 * dx, dx)

    t_lower = 200.0
    t_upper = 1000.0

    nPts = len(x)
    
    Q = np.zeros(nPts)
    bool_1 = np.array([x>3])
    bool_2 = np.array([x<7])
    # or: np.logical_and(x>3, x<7)
    # np.logical_and(x>3, x<7) == (bool_1 & bool_2).squeeze()
    Q[(bool_1 & bool_2).squeeze()] = 100
    lambda_var = 100
    

    # set default coefficients for the solver:
    a = np.zeros(nPts) + 1
    b = np.zeros(nPts) - 2
    c = np.zeros(nPts) + 1
    d = np.zeros(nPts) - Q/lambda_var * dx**2

    # boundary conditions (bottom - fixed):
    a[0] = 0
    b[0] = 1
    c[0] = 0
    d[0] = t_lower

    # top - fixed:
    a[-1] = 1       # 1     0
    b[-1] = -1       # -1    1
    c[-1] = 0       # 0     0
    d[-1] = 0       # 0     t_upper

    # Add a source term:
    
    # solve for Temperature:
    t = solve_tridiagonal(a, b, c, d)

    # plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)

    ax.plot(x, t)

    plotfile = 'conduction_v1.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    
    
