"""
CALCULATING N PROFILE FROM TEMP PROFILE
"""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
import matplotlib.pyplot as plt


def temp_to_dens(n0 = 1e19, alt0 = 100, altn = 500, nPts = 500, r = 6370, m = 28*1.67e-27, k = 1.38e-23,
                 T0 = 200, Tn = 1000):
    size_of_steps = ((altn-alt0)/nPts)
    size_of_steps_temp = ((Tn-T0)/nPts)
    nPtsAr = np.arange(1,nPts+1)
    alt = np.linspace(alt0, altn, nPts)
    T = np.linspace(T0, Tn, nPts)
    g = np.array([3.99e14 / ((r+i)*1000)**2 for i in alt])
    temp=(T[1:]+T[:-1])/2
    gravity=(g[1:]+g[:-1])/2
    H = k*temp/m/gravity
    n = [n0]
    for h, t_0, t_1, dz in zip(H, T[:-1], T[1:], (alt[1:]-alt[:-1])*1000):
            n += [t_0/t_1 * n[-1] * np.exp(-1*dz/h)]
    return(n,alt,H)
    
if __name__ == "__main__":   
    
    nn, altn, Hn = temp_to_dens()
    
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(altn, nn)
    plt.title("density as a function of altitude in the thermosphere", size=14, fontfamily='georgia')
    ax.set_ylabel("density value [kg/m^3]", size=14, fontfamily='georgia')
    ax.set_xlabel("altitude [km]", size=14, fontfamily='georgia')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(altn, nn)
    ax.set_yscale('log')
    plt.title("density as a function of altitude in the thermosphere", size=14, fontfamily='georgia')
    ax.set_ylabel("density value log[kg/m^3]", size=14, fontfamily='georgia')
    ax.set_xlabel("altitude [km]", size=14, fontfamily='georgia')
    plt.show()
    
    plt.plot(Hn)
    
    # N2: 28*1e19; O2: 32*0.3e19; O:16*1e18 
    
    