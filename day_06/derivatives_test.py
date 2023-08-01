#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Take first derivative of a function
# ----------------------------------------------------------------------

def first_derivative(f, x):

    """ Function that takes the first derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    dfdx - the first derivative of f(x)

    Notes
    -----
    take the first derivative of f(x) here
    
    # (f(x+dx) - f(x-dx)) / 2dx

    """
    
    nPts = len(f)
    
    dfdx = np.zeros(nPts)
    #for i in range(1,nPts-1):
    #    dfdx[i] = (f[i+1]-f[i-1])/abs(x[i+1]-x[i-1])
    dfdx[1:-1] = (f[2:]-f[:-2])/abs(x[2:]-x[:-2])
    dfdx[0] = (-3*f[0] + 4*f[1] - f[2]) / abs(x[2]-x[0]) #float("NaN") 
    dfdx[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / abs(x[-1]-x[-3]) # float("NaN")
    
    # do calculation here - need 3 statements:
    #  1. left boundary ( dfdx(0) = ...)
    #  2. central region (using spans, like dfdx(1:nPts-2) = ...)
    #  3. right boundary ( dfdx(nPts-1) = ... )

    return dfdx

# ----------------------------------------------------------------------
# Take second derivative of a function
# ----------------------------------------------------------------------

def second_derivative(f, x):

    """ Function that takes the second derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    take the second derivative of f(x) here
    
    # (f(x+dx) + f(x-dx)) - 2f(x) / dx^2
    """
    from numpy import power
    nPts = len(f)
    
    d2fdx2 = np.zeros(nPts)

    #for i in range(1,nPts-1):
    #    d2fdx2[i] = (f[i+1]+f[i-1]-2*f[i])/power((x[i+1]-x[i]),2)
    d2fdx2[1:-1] = (f[2:]+f[:-2]-2*f[1:-1])/power((x[2:]-x[1:-1]),2)
    d2fdx2[0] =  (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / power(abs(x[0]-x[1]),2) # float("NaN")
    d2fdx2[-1] = (2*f[-1] - 5*f[-2] + 4*f[-3] - f[-4]) / power(abs(x[0]-x[1]),2)    #float("NaN") 
    
    # do calculation here - need 3 statements:
    #  1. left boundary ( dfdx(0) = ...)
    #  2. central region (using spans, like dfdx(1:nPts-2) = ...)
    #  3. right boundary ( dfdx(nPts-1) = ... )

    return d2fdx2

# ----------------------------------------------------------------------
# Get the analytic solution to f(x), dfdx(x) and d2fdx2(x)
# ----------------------------------------------------------------------

def analytic(x):

    """ Function that gets analytic solutions

    Parameters
    ----------
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    f - the function evaluated at x
    dfdx - the first derivative of f(x)
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    These are analytic solutions!

    """
    from numpy import cos, sin
    from numpy import power

    #f = 4 * x ** 2 - 3 * x -7
    #dfdx = 8 * x - 3
    #d2fdx2 = np.zeros(len(f)) + 8.0
    
    f = 3 * sin(x) + 5 * power(x,2) - power(x,3)
    #fctstring = "3sin(x) + 5x^2 - x^3"
    
    dfdx = 3* (cos(x)) + 10*x - 3*power(x,2)
    #dxstring = '3cos(x) + 10x - 3x^2'
    
    d2fdx2 = 3*-(sin(x)) + 10 - 6*x
    #dx2string = '-3sin(x)+10-6x'

    return f, dfdx, d2fdx2 #, fctstring, dxstring, dx2string

def integration(f, x):
    
    nPts = len(f)
    
    integral = np.zeros(nPts)
    integral[:-1] = ((f[:-1]+f[1:])/2*abs(x[:-1]-x[1:]))
    integral[-1] = (f[-1]+f[-2])*abs(x[-2]-x[-1])
    
    return integral

def analytic_int(x):
    
    from numpy import cos, sin
    from numpy import power  
    
    f =  3 * sin(x) + 5 * power(x,2) - power(x,3)
    
    int_f = -3*cos(x) + 5/3 * power(x,3) - 1/4*power(x,4)

    return f, int_f

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    route = r"C:/Users/Niki/Dokumentumok/GIT/swsss2023"

    # define dx:
    dx = np.pi / 2
    
    # arange doesn't include last point, so add explicitely:
    x = np.arange(-3.0 * np.pi, 3.0 * np.pi + dx, dx)

    # get analytic solutions:
    f, a_dfdx, a_d2fdx2 = analytic(x) #, fstr, dxstr, dx2str

    # get numeric first derivative:
    n_dfdx = first_derivative(f, x)

    # get numeric first derivative:
    n_d2fdx2 = second_derivative(f, x)

    # plot:
        
    fig, ax = plt.subplots(nrows=3, ncols = 1, figsize = (10,7), dpi=100)
    ax[0].plot(x, f) #, label=r'${0}$'.format(fstr)
    ax[0].legend()
    # plot first derivatives:
    error1 = np.sum(np.abs(n_dfdx - a_dfdx)) / (len(n_dfdx))
    sError1 = " (Err: %0.8f)" % error1
    ax[1].plot(x, a_dfdx, color = 'black', label = 'Analytic') #  (${0}$)'.format(dxstr)
    ax[1].plot(x, n_dfdx, color = 'red', label = 'Numeric (Err: {0:.5f})'.format(error1))
    ax[1].scatter(x, n_dfdx, color = 'red')
    ax[1].legend()

    # plot second derivatives:
    error2 = np.sum(np.abs(n_d2fdx2 - a_d2fdx2)) / (len(n_d2fdx2))
    sError2 = " (Err: %0.8f)"  % error2
    ax[2].plot(x, a_d2fdx2, color = 'black', label = 'Analytic') # (${0}$)'.format(dx2str)
    ax[2].plot(x, n_d2fdx2, color = 'red', label = 'Numeric (Err: {0:.5f})'.format(error2))
    ax[2].scatter(x, n_d2fdx2, color = 'red')
    ax[2].legend()

    plotfile = r'/plot3.png'
    print('writing : ',route+plotfile)    
    fig.savefig(route+plotfile)
    plt.show()
    plt.close()
    
    
