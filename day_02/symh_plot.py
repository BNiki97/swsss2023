"""SYM/H index reader and plotter code."""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import DateFormatter

def read_ascii_file(filename,index):    
    """
    function that reads OMNI SYM/H data
    INPUT:
        name of file
        number of line to start gathering data from
    OUTPUT:
        dictionary containing the time ('time') and symh ('symh')
    """
    year = []
    doy = []
    hour = []
    minute = []
    temp = []
    data = {"time":[],
       "symh":[]}
    to_delete = []
    with open(filename) as f:
        for i in range(index):
            to_delete=(f.readline())
        for line in f:
            temp=(line.split())
            year=(int(temp[0]))
            doy=(int(temp[1]))
            hour=(int(temp[2]))
            minute=(int(temp[3]))
            data["time"].append(dt.datetime(year,1,1) + dt.timedelta(days = doy-1,hours=hour,minutes=minute))
            data['symh'].append(float(temp[4]))
    # part 3: return results
    return(data)

import matplotlib.pylab as pylab
fontsizer=20
params = {'axes.labelsize': fontsizer,
          'axes.titlesize':fontsizer,
          'xtick.labelsize':fontsizer,
          'ytick.labelsize':fontsizer,
        'figure.figsize': (20, 10),
         'font.family':'arial'}
pylab.rcParams.update(params)

test = r"C:\Users\Niki\Documents\GIT\swsss2023\day_02\omni_test.lst"
route = r"C:\Users\Niki\Documents\GIT\swsss2023\day_02\omni_min_def_67ewb9WLYP.lst"

data_dict=read_ascii_file(route,0)

fig,ax = plt.subplots()
ax.plot(data_dict['time'][:],data_dict['symh'][:],'.', label='dataset')
ax.set(title="Geomagnetic Storm in 2013 March 17")
ax.set_xlabel("Date")
ax.set_ylabel("SYM-H index")
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.grid()
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.show()

   
    
    