"""SYM/H index reader and plotter code."""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import DateFormatter

import os.path

def read_ascii_file(filename,index,starttime=dt.datetime(1950,1,1),endtime=dt.datetime(2050,1,1)):    
    """
    function that reads OMNI SYM/H data
    INPUT:
        name of file
        the column number of data (must be integer)
        starting time for data reading (default: 1900.01.01)
        ending time for data reading (default: 2050.01.01)
    OUTPUT:
        dictionary containing the time ('time') and symh ('symh')
    """
    assert (os.path.isfile(filename)), "file doesn't exist"
    assert isinstance(starttime,dt.datetime), "check starttime input"
    assert isinstance(endtime,dt.datetime), "check endtime input"  
    assert isinstance(index,int), "check index"
    year = []
    doy = []
    hour = []
    minute = []
    temp = []
    data = {"time":[],
       "symh":[]}
    with open(filename) as f:
        for line in f:
            temp=(line.split())
            year=(int(temp[0]))
            doy=(int(temp[1]))
            hour=(int(temp[2]))
            minute=(int(temp[3]))
            data["time"].append(dt.datetime(year,1,1) + dt.timedelta(days = doy-1,hours=hour,minutes=minute))
            data['symh'].append(float(temp[index]))
            
    time = np.array(data['time'])
    indexer = (time>starttime)&(time<endtime)
    time_selected = time[indexer]
    symh = np.array(data['symh'])
    symh_selected = symh[indexer]
    
    data['symh'] = symh_selected
    data['time'] = time_selected
    # part 3: return results
    return(data)

import matplotlib.pylab as pylab
fontsizer=22
params = {'legend.fontsize': 20,
          'legend.markerscale': 4,
          'axes.labelsize': fontsizer,
          'axes.titlesize':fontsizer,
          'xtick.labelsize':fontsizer,
          'ytick.labelsize':fontsizer,
        'figure.figsize': (20, 10),
         'font.family':'georgia'}
pylab.rcParams.update(params)


if __name__ == '__main__':  # main code block
    test = r"C:\Users\Niki\Documents\GIT\swsss2023\day_02\omni_test.lst"
    route = r"C:\Users\Niki\Documents\GIT\swsss2023\day_02\omni_min_def_67ewb9WLYP.lst"
    
    data_dict=read_ascii_file(route,-1) #,dt.datetime(2013,3,16),dt.datetime(2013,3,21)
    
    # select data lower than -100 T
    symh = np.array(data_dict['symh'])
    indexer = (symh<-100)
    time = np.array(data_dict['time'])
    symh_selected = symh[indexer] 
    time_selected = time[indexer]
    
    # plot
    fig,ax = plt.subplots()
    ax.plot(data_dict['time'][:],data_dict['symh'][:],'.', label='dataset')
    ax.plot(time_selected,symh_selected,'r.',label='measurements below -100 nT')
    ax.set(title="Geomagnetic Storm in 2013 March 17")
    ax.set_xlabel("Date")
    ax.set_ylabel("SYM-H index")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.show()

    
    
    