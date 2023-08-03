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

    dx = 4.0
    nDays = 50
    dt = 0.5
    
    # set x with 1 ghost cell on both sides:
    x = 100+np.arange(-dx, 400 + 2*dx, dx) #np.arange(-dx, 10 + 1 * dx, dx) #np.arange(-dx, 10 + 2 * dx, dx) #alt = 100+40*x
    
    nPts = len(x)
    lon = 110.0
    
    LT = np.arange(0,nDays*24,dt)
    f107 = 100 + 50/(24*365)*LT + 25*np.sin(LT/(24*27)*2*np.pi)
        
    lambda_var = 80    
    ampDi = 10
    ampLo = 10
    ampSDi = 5
    phDi = np.pi/2
    phDSi = 3*np.pi/2
    phLo = np.pi/4
    
    x_lower = 200
    x_upper = 400

    #bool_1 = np.array([x>3])
    #bool_2 = np.array([x<7])
    # or: np.logical_and(x>3, x<7)
    # np.logical_and(x>3, x<7) == (bool_1 & bool_2).squeeze()
    Q_background = np.zeros(nPts)
    Q_euv = np.zeros(nPts)
    Q_background[np.logical_and(x>x_lower, x<x_upper)] = 0.4
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
    loc_time = []
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    for i, timeCurr in enumerate(LT):
        ut = timeCurr % 24
        local_time = lon/15.0 + ut
        if local_time > 24:
            local_time = local_time-24
        #LT[i] = local_time
        t_lower = 200.0 + ampDi*np.sin(local_time/24*2*np.pi+phDi) + ampSDi*np.sin(local_time/24*4*np.pi+phDSi) + ampLo*np.sin(lon/360*2*np.pi+phLo)
        fac = np.array(-np.cos(local_time/24*2*np.pi))
        fac[(fac < 0)] = 0
        SunHeat = f107[i] * 0.4/100
        Q_euv[(np.logical_and(x>x_lower, x<x_upper))] = SunHeat * fac
        d = np.zeros(nPts) - (Q_background+Q_euv)/lambda_var * dx**2
        d[0] = t_lower
        d[-1] = 0
        t.append(solve_tridiagonal(a, b, c, d))
        loc_time.append(local_time)
        ax.plot(x, t[i])
    plotfile = 'conduction_v2_full.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    fig,ax = plt.subplots(figsize=(16,8), dpi=200)
    heh=ax.contourf(LT/24,x,np.array(t).T) #alt
    ax.set_xlabel("time [days]")
    ax.set_ylabel("altitude [km]")
    cbar = fig.colorbar(heh)
    cbar.ax.set_ylabel("Temperature [K]")
    plt.title("thermosphere temperature conditions as a function of altitude and local time\nfor the longitude of {0} degrees".format(lon))
    plotfile = 'conduction_v2_contour50d_{0}.png'.format(int(lon))
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.show()
    plt.close()


    
    
