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
    nDays = 3
    dt = 0.25
    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10 + 2 * dx, dx) #np.arange(-dx, 10 + 2 * dx, dx)
    alt = 100+40*x
    nPts = len(x)
    LT = np.arange(0,nDays*24,dt)
    lon = 21.0

    t_lower = 200.0
    lambda_var = 10
    SunHeat = 100

    #bool_1 = np.array([x>3])
    #bool_2 = np.array([x<7])
    # or: np.logical_and(x>3, x<7)
    # np.logical_and(x>3, x<7) == (bool_1 & bool_2).squeeze()
    Q_background = np.zeros(nPts)
    Qeuv = np.zeros(nPts)

    Q_background[np.logical_and(x>3, x<7)] = 100
    a = np.zeros(nPts) + 1
    b = np.zeros(nPts) - 2
    c = np.zeros(nPts) + 1

    a[0] = 0
    b[0] = 1
    c[0] = 0

    a[-1] = 1
    b[-1] = -1
    c[-1] = 0

    t=[]
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    for i, timeCurr in enumerate(LT):
        ut = timeCurr % 24
        #print(ut)
        local_time = lon/15.0 + ut
        fac = np.array(-np.cos(local_time/24*2*np.pi))
        fac[(fac < 0)] = 0
        #print(local_time)
        Qeuv[(np.logical_and(x>3, x<7))] = SunHeat * fac
        d = np.zeros(nPts) - (Q_background+Qeuv)/lambda_var * dx**2
        d[0] = t_lower
        d[-1] = 0
        t.append(solve_tridiagonal(a, b, c, d))
        ax.plot(x, t[i])
    plotfile = 'conduction_v2_full.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    fig,ax = plt.subplots(figsize=(12,8))
    heh=ax.contourf(LT/24,alt,np.array(t).T)
    ax.set_xlabel("local time [days]")
    ax.set_ylabel("altitude [km]")
    cbar = fig.colorbar(heh)
    cbar.ax.set_ylabel("Temperature [K]")
    plt.title("thermosphere temperature conditions as a function of altitude and local time\nfor the longitude of {0} degrees".format(lon))
    plotfile = 'conduction_v2_contour_{0}.png'.format(int(lon))
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.show()
    plt.close()

    
    
