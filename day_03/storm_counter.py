"""SYM/H index reader and plotter code."""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import numpy as np
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
    route = r"C:\Users\Niki\Documents\GIT\swsss2023\day_03\symh_2003.lst"
    
    data_dict=read_ascii_file(route,-1) #,dt.datetime(2013,3,16),dt.datetime(2013,3,21)
    
    fig,ax = plt.subplots()
    ax.plot(data_dict['time'][:],data_dict['symh'][:],'.', label='dataset')
    #ax.plot(time_selected,symh_selected,'r.',label='measurements below -100 nT')
    ax.set(title="Geomagnetic Storm in 2013 March 17")
    ax.set_xlabel("Date")
    ax.set_ylabel("SYM-H index")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.show()
    
    symh = np.array(data_dict['symh'])
    indexer = (symh<-100)
    time = np.array(data_dict['time'])
    symh_selected = symh[indexer] 
    time_selected = time[indexer]

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
    
    
    
    
    start_indices = []
    end_indices = []
    
    for i in range(len(symh)-1):
        if symh[i] >= -100 and symh[i+1] < -100:
            start_indices.append(i)
        elif symh[i] < -100 and symh[i+1] >= -100:
            end_indices.append(i)
    storm_commencement = []
    storm_ending = []
    index=0
    for i in range(len(start_indices)-1):
        if i == index:            
            for j in range(i+1,len(start_indices)):
                if abs(time[start_indices[i]] - time[start_indices[j]]) > dt.timedelta(hours=12):
                    storm_commencement.append(start_indices[i])
                    storm_ending.append(end_indices[j-1])
                    index = j
                    break
    storm_commencement.append(start_indices[index])
    storm_ending.append(end_indices[-1])
    
                
    index_helper = []
    peaks = []
    for i in range(0,len(storm_commencement)):
        temporary_array = list(symh[storm_commencement[i]:storm_ending[i]])
        #plt.plot(temporary_array)
        if len(temporary_array) >= 5:
            peaks.append(temporary_array.index(min(temporary_array)))
            index_helper.append(storm_commencement[i])
    for i in range(len(peaks)):
        peaks[i] = peaks[i] + index_helper[i]
        
        
    # there are outliners, making a running average
    window_size = 6
    i=0
    symh_moving_averages = []
    time_moving_averages = time[int(window_size/2):-int(window_size/2)+1]
    while i < len(symh) - window_size + 1:    
        # Store elements from i to i+window_size
        # in list to get the current window
        window = symh[i : i + window_size]
  
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
      
        # Store the average of current
        # window in moving average list
        symh_moving_averages.append(window_average)
      
        # Shift window to right by one position
        i += 1  
    plt.plot(time_moving_averages,symh_moving_averages,'.')
    

    symh_moving_array = np.array(symh_moving_averages)
    time_moving_array = np.array(time_moving_averages)
    indexer = (symh_moving_array<-100)
    symh_selected = symh_moving_array[indexer] 
    time_selected = time_moving_array[indexer]
    
    find_timegaps = []
    for index_number in range(len(time_selected)-1):
        if abs(time_selected[index_number] - time_selected[index_number+1]) > dt.timedelta(hours = 12):
        #if (symh_selected[index_number] - symh_selected[index_number+1]) > 10:
            find_timegaps.append(index_number)
    find_timegaps_new = []
    find_timegaps_new.append(0)
    for i in find_timegaps:
        find_timegaps_new.append(i)
    find_timegaps_new.append(-1)
    peaks = []
    for i in range(1,len(find_timegaps_new)):
        temporary_array = list(symh_selected[find_timegaps_new[i-1]:find_timegaps_new[i]])
        plt.plot(temporary_array)
        peaks.append(temporary_array.index(min(temporary_array)))
    for i in range(len(find_timegaps_new)-1):
        peaks[i] = peaks[i] + find_timegaps_new[i]
    
    fig,ax = plt.subplots()
    ax.plot(data_dict['time'][:],data_dict['symh'][:],'.', label='dataset')
    ax.plot(time_selected,symh_selected,'g.',label='measurements below -100 nT')
    ax.plot(time[peaks], symh[peaks], "rx",label='here is a storm!')
    ax.set(title="Geomagnetic Storm in 2013 March 17\nNUMBER OF STORMS: {0}".format(len(peaks)))
    ax.set_xlabel("Date")
    ax.set_ylabel("SYM-H index")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.grid()
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.show() 

"""
# DISCONTINUED
    from scipy.signal import find_peaks
    symh_selected_new = [-x for x in symh_selected]
    peaks, _ = find_peaks(symh_selected_new, prominence=10)
    
    plt.plot(time_selected,symh_selected, '.')
    plt.plot(time_selected[peaks],symh_selected[peaks],'rx')
"""


    
    
    