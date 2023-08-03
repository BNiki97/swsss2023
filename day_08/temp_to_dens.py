"""
CALCULATING N PROFILE FROM TEMP PROFILE
"""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
import matplotlib.pyplot as plt

def calc_grav(r,alt):
    return(np.array([3.99e14 / ((r+i)*1000.0)**2 for i in alt]))

def calc_scaleheight(k,temp,m,gravity):
    return k*temp/m/gravity

def temp_to_dens(n0 = 1e19, alt0 = 100, altn = 500, nPts = 500, r = 6370, m = 28*1.67e-27, k = 1.38e-23,
                 T0 = 200, Tn = 1000, Tin = []):
    
    alt = np.linspace(alt0, altn, nPts)
    
    if len(Tin) == 0:
        print("building own Temp array")
        T = np.linspace(T0, Tn, nPts)
    else:
        T = Tin

    temp = []
    for i in range(nPts-1):
        temp.append((T[i+1]+T[i])/2)
    temp = np.array(temp) #temp = np.array(T) #
    
    g = calc_grav(r,alt)
    gravity = []
    for i in range(nPts-1):
        gravity.append((g[i+1]+g[i])/2)
    gravity=np.array(gravity) #gravity=np.array(g) #
    H = calc_scaleheight(k,temp,m,gravity)
    
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
    
    