#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal
from temp_to_dens import temp_to_dens

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

# smoother equation - this is what conduction does
# a weighted average between the two neighboring values

def temperature_calculator(nPts, x, LT, lon=21, QB_value= 0.4, x_lower = 200, x_upper = 400, lambda_var = 80, ampDi=10, ampLo = 10, ampSDi = 5, phDi = np.pi/2, phDSi = 3*np.pi/2, phLo = np.pi/4):
    
    f107 = 100 + 50/(24*365)*LT + 25*np.sin(LT/(24*27)*2*np.pi)  
    Q_background = np.zeros(nPts)
    Q_background[np.logical_and(x>x_lower, x<x_upper)] = QB_value
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
    for i, timeCurr in enumerate(LT):
        Q_euv = np.zeros(nPts)
        ut = timeCurr % 24
        local_time = lon/15.0 + ut
        if local_time > 24:
            local_time = local_time-24
        t_lower = 200.0 + ampDi*np.sin(local_time/24*2*np.pi+phDi) + ampSDi*np.sin(local_time/24*4*np.pi+phDSi) + ampLo*np.sin(lon/360*2*np.pi+phLo)
        fac = np.array(-np.cos(local_time/24*2*np.pi))
        fac[(fac < 0)] = 0
        SunHeat = f107[i] * 0.4/100
        Q_euv[(np.logical_and(x>x_lower, x<x_upper))] = SunHeat * fac
        d = np.zeros(nPts) - (Q_background+Q_euv)/lambda_var * abs(x[0]-x[1])**2
        d[0] = t_lower
        d[-1] = 0
        t.append(solve_tridiagonal(a, b, c, d))
    fig,ax = plt.subplots(figsize=(16,8), dpi=200)
    heh=ax.contourf(LT/24,x,np.array(t).T) #alt
    ax.set_xlabel("time [days]")
    ax.set_ylabel("altitude [km]")
    cbar = fig.colorbar(heh)
    cbar.ax.set_ylabel("Temperature [K]")
    plt.title("thermosphere temperature conditions as a function of altitude and local time\nfor the longitude of {0} degrees".format(lon))
    plotfile = 'conduction_v2_contour_{0}.png'.format(int(lon))
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.show()
    plt.close()
    return(t)

def calc_dens(t,nPts,m=28*1.67e-27,n0 = 1e19):
    n = []
    for i in range(len(t)):
        temporary, ti, to = temp_to_dens(n0 = n0, nPts = nPts, m=m, Tin=t[i])
        n.append(temporary)
    return(n)

def plot_dens(key_arr,dict_of_val,LT,x):
    from matplotlib import ticker    
    for key in key_arr:
        fig,ax = plt.subplots(figsize=(16,8), dpi=200)
        heh=ax.contourf(LT/24,x,(dict_of_val[key]).T, locator=ticker.LogLocator()) #alt
        ax.set_xlabel("time [days]")
        ax.set_ylabel("altitude [km]")
        cbar = fig.colorbar(heh)
        cbar.ax.set_ylabel("log density [kg/m^3]")
        plt.title("density of {0} in the thermosphere as a function of altitude and local time\nfor the longitude of {1} degrees".format(key,lon))
        plotfile = 'conduction_v2_contour50d_{0}_{1}.png'.format(int(lon),key)
        print('writing : ',plotfile)    
        fig.savefig(plotfile)
        plt.show()
        plt.close()
        
def temperature_calculator_timedep(dx, dt, nPts, nDays=1, QB_value= 0.4, x_lower = 200, x_upper = 400, lambda_var = 80, value_for_sunheat=0.4/100, ampDi=10, ampLo = 10, ampSDi = 5, phDi = np.pi/2, phDSi = 3*np.pi/2, phLo = np.pi/4):
    x = 100+np.arange(-dx, 400 + 2*dx, dx)
    LT = np.arange(0,nDays*3600*24,dt)
    f107 = 100 + 50/(3600*24*365)*LT + 25*np.sin(LT/(3600*24*27)*2*np.pi)  
    Q_background = np.zeros(nPts)
    Q_background[np.logical_and(x>x_lower, x<x_upper)] = QB_value
    #lambda_var = 1.0e5 + 100*(x-100.0)**2
    k=dt*lambda_var/(dx**2)
    a = np.zeros(nPts) - k
    b = np.zeros(nPts) + 1+2*k
    c = np.zeros(nPts) - k
    d = np.zeros(nPts)
    a[-1] = 1
    b[-1] = -1
    c[-1] = 0
    a[0] = 0
    b[0] = 1
    c[0] = 0
    t=[]
    for i, timeCurr in enumerate(LT):
        Q_euv = np.zeros(nPts)
        ut = (timeCurr) % (24*3600)
        #print(ut)
        local_time = lon/15.0 + ut
        #print(local_time)
        if local_time > (24*3600):
            local_time = local_time-(24*3600)
        t_lower = 200.0 + ampDi*np.sin(local_time/24/3600*2*np.pi+phDi) + ampSDi*np.sin(local_time/24/3600*4*np.pi+phDSi) + ampLo*np.sin(lon/360*2*np.pi+phLo)
        fac = np.array(-np.cos(local_time/24/3600*2*np.pi))
        fac[(fac < 0)] = 0
        SunHeat = f107[i] * value_for_sunheat
        Q_euv[(np.logical_and(x>x_lower, x<x_upper))] = SunHeat * fac
        if i == 0:
            d = t_lower + dt*((Q_background+Q_euv)/lambda_var * abs(x[0]-x[1])**2)
            d[0] = t_lower
            d[-1] = 0
        else:
            d = t[i-1] + dt*((Q_background+Q_euv)/lambda_var * abs(x[0]-x[1])**2)
            d[0] = t_lower
            d[-1] = 0
        t.append(solve_tridiagonal(a, b, c, d))
    fig,ax = plt.subplots(figsize=(16,8), dpi=200)
    heh=ax.contourf(LT/3600/24,x,np.array(t).T) #alt
    ax.set_xlabel("time [days]")
    ax.set_ylabel("altitude [km]")
    cbar = fig.colorbar(heh)
    cbar.ax.set_ylabel("Temperature [K]")
    plt.title("thermosphere temperature conditions as a function of altitude and time\nfor the longitude of {0} degrees".format(lon))
    plotfile = 'conduction_v2_contour_timedep_{0}.png'.format(int(lon))
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.show()
    plt.close()
    return t

"""
        temporary, ti, to = temp_to_dens(n0 = 1e19, nPts = nPts, m=4.65e-26, Tin=t[i])
        n_N2.append(temporary)
        temporary, ti, to = temp_to_dens(n0 = 1e19, nPts = nPts, m=2*2.6566962e-26, Tin=t[i])
        n_O2.append(temporary)
        temporary, ti, to = temp_to_dens(n0 = 1e19, nPts = nPts, m=2.6566962e-26, Tin=t[i])
        n_O.append(temporary)
        loc_time.append(local_time)
"""

if __name__ == "__main__":

    dx = 4
    nDays = 27
    dt = 0.25
    dt_new = 10
    LT = np.arange(0,nDays*24,dt)
    x_lower = 200
    x_upper = 400,
    # set x with 1 ghost cell on both sides:
    x = 100+np.arange(-dx, 400 + 2*dx, dx) #np.arange(-dx, 10 + 1 * dx, dx) #np.arange(-dx, 10 + 2 * dx, dx) #alt = 100+40*x    
    nPts = len(x)
    lon = 22.0    
    
    t=temperature_calculator(nPts,x,LT,lon=lon,x_lower=x_lower,x_upper=x_upper)
    t_prob = temperature_calculator_timedep(dx,dt_new,nPts,nDays=27) #,  ,value_for_sunheat=1/2,QB_value=10
    
    amu = 1.67377e-27
    keys = ['N2', 'O', 'O2', 'CO2']
    
    dict_of_masses = {keys[0]: 28.2*amu,
                      keys[1]: 16*amu,
                      keys[2]: 2*16*amu,
                      keys[3]: 44*amu}
    dict_of_values = {keys[0]: [],
                      keys[1]: [],
                      keys[2]: [],
                      keys[3]: []}
    for key in keys:
        dict_of_values[key].append(calc_dens(t,nPts=nPts,m=dict_of_masses[key]))
        dict_of_values[key]=np.array(dict_of_values[key]).squeeze()
    
    plot_dens(keys,dict_of_values,LT,x)
    
    
    

    
