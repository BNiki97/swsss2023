"""
PROJECT 4 FOR DAY 5 OF SWSSS2023.
"""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import datetime
import matplotlib.pyplot as plt
import h5py

def extract_data(dictionary, key):
    """
    helper function, you don't need it
    """
    return(dictionary.__dict__[key])

def champ_reader(files):
    """
    Reads in CHAMP data that are contained in files.

    Parameters
    ----------
    files : routes to CHAMP datafiles

    Returns
    df: dataframe containing the data
    header: header of the df

    """
    with open(files[0]) as f: #route + r"/" + files[0]
        header = ((f.readline().split(',')))   
        header[-1] = header[-1].split('\n')[0]
    class NewClass(object): pass
    dictionary = NewClass()
    for var in header: 
        setattr(dictionary, var, []) 
    for number, file in enumerate(files):
        with open(file) as f:
            to_delete = ((f.readline().split(',')))
            for line in f:
                temp=(line.split())
                if float(temp[-3]) < -6e-10:
                    temp[-3] = 0
                for j, item in enumerate(temp):
                    dictionary.__dict__[header[j]].append(float(item))
                dictionary.__dict__[header[0]][-1] = datetime.datetime(2002,1,1) + datetime.timedelta(days = number, seconds = float(temp[0]))
    df = pd.DataFrame(dictionary.__dict__)
    return(df, header)

def create_hourly_data(df, header, num_to_divide_by = 3600):
    """
    Takes every hour's value. No averaging, just extracting data at every *:00:00 timestep.

    Parameters
    ----------
    df : dataframe returned by champ_reader
    header : header returned by champ_reader
    num_to_divide_by : TYPE, optional
        DESCRIPTION. The default is 3600, which is seconds in hour.
        If you want data every two hours, make it 7200... etc.
    Returns
    -------
    df: dataframe of hourly data (header will be same as the input df)

    """
    class NewClass(object): pass
    dictionary_hourly = NewClass()
    for var in header: 
        setattr(dictionary_hourly, var, [])
    for index in range(len(df[header[0]])):
        if ((df[header[0]][index] - datetime.datetime(1980,1,1)).total_seconds() % num_to_divide_by) == 0:
            for item in header:
                dictionary_hourly.__dict__[item].append(df[item][index])
    df_hourly = pd.DataFrame(dictionary_hourly.__dict__)
    return(df_hourly)
    
def interpolate_density(model_file, df_hourly, header, model = 'jb2008'):
    """
    Parameters
    ----------
    model_file : the route to the model file
    df_hourly : dataframe returned by create_hourly_data
    header : returned by champ_reader, the header for the df
    model : TYPE, optional
        DESCRIPTION. The default is 'jb2008'.
        You get to set whether you want the jb2008 or the tiegcm
    Returns
    -------
    one variable: the array containing the interpolated data for every time step of df_hourly

    """

    assert (model == 'jb2008' or model == 'tiegcm'), 'model not valid'
    if model == 'jb2008':
        try:
            loaded_data = loadmat(model_file)
        except:
            print("File not found. Please check your directory")
        JB2008_dens = loaded_data['densityData']
        localSolarTimes_JB2008 = np.linspace(0,24,24)
        latitudes_JB2008 = np.linspace(-87.5,87.5,20)
        altitudes_JB2008 = np.linspace(100,800,36)
        nofAlt_JB2008 = altitudes_JB2008.shape[0]
        nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
        nofLat_JB2008 = latitudes_JB2008.shape[0]
        JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008, nofAlt_JB2008,8760), order='F') # Fortran-like index order        
    elif model == 'tiegcm':
        loaded_data = h5py.File(model_file)
        tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
        altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
        latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
        localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
        nofAlt_tiegcm = altitudes_tiegcm.shape[0]
        nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
        nofLat_tiegcm = latitudes_tiegcm.shape[0]
        tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
    interpolated_density_4champ = []

    time_start = 0
    time_end = len(df_hourly[header[0]])
    for i in range(time_start,time_end):
        time_index_int = int((df_hourly[header[0]][i] - datetime.datetime(2002,1,1)).total_seconds()/3600)
        if model == 'jb2008':
            sample_data = JB2008_dens_reshaped[:,:,:,time_index_int]
            x = localSolarTimes_JB2008 
            y = latitudes_JB2008
            z = altitudes_JB2008   
        elif model == 'tiegcm':
            sample_data = tiegcm_dens_reshaped[:,:,:,time_index_int]
            x = localSolarTimes_tiegcm 
            y = latitudes_tiegcm
            z = altitudes_tiegcm
        interpolated_function = RegularGridInterpolator((x,y,z), sample_data, bounds_error=False, fill_value=None)
        xx = df_hourly[header[4]][i]
        yy = df_hourly[header[2]][i]
        zz = df_hourly[header[1]][i]
        X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
        interpolated_density_4champ.append(interpolated_function((X, Y, Z)).squeeze())
    return(interpolated_density_4champ)

def plot_density_data(interpolated_density_4champ, time_array, name_modifier = 1, figsize=(12,5)):
    """
    plots the interpolated density data
    interpolated_density_4champ: that returned by interpolate_density
    time_array: the datetimes, take it from the df_hourly
    name_modifier: based on this the title will change, 1: jb2008 model, 2: tie-gcm model, 3: accelerometer data
    
    returns fig, ax if needed
    """
    labels = ['jb2008 model', 'tie-gcm model', 'derived parameter']
    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize = figsize)
    ax.plot(time_array,interpolated_density_4champ, '--', label=labels[name_modifier-1]) #, '.', ms = 5
    ax.set_ylabel("density value [kg/m^3]", size=14, fontfamily='georgia')
    ax.set_xlabel("time [YYYY-MM-DD]", size=14, fontfamily='georgia')
    if name_modifier == 1 or name_modifier==2:
        plt.title('interpolated density values for CHAMP trajectory', fontfamily='georgia', size=16)
    elif name_modifier == 3:
        plt.title('accelerometer-derived density values for CHAMP trajectory', fontfamily='georgia', size=16)
    plt.legend()
    plt.plot()
    return fig, ax

def compare_density_data(interpolated_density_4champ_jb, interpolated_density_4champ_tie, time_array, figsize=(12,5)):
    """
    compares the interpolated density data
    interpolated_density_4champ_jb: that returned by interpolate_density for jb2008 model
    interpolated_density_4champ_tie: that returned by interpolate_density for tie-gcm model
    time_array: the datetimes, take it from the df_hourly
    
    returns fig, ax if needed
    """
    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize = figsize)
    ax.plot(time_array,interpolated_density_4champ_jb, '.-', label='jb2008 model') #, '.', ms = 5
    ax.plot(time_array,interpolated_density_4champ_tie, '--', label='tie-gcm model') #, '.', ms = 5
    ax.set_ylabel("density value [kg/m^3]", size=14, fontfamily='georgia')
    ax.set_xlabel("time [YYYY-MM-DD]", size=14, fontfamily='georgia')
    plt.title('comparison of interpolated density values for CHAMP trajectory', fontfamily='georgia', size=16)
    plt.legend()
    plt.plot()
    return fig, ax

def plot_all_data(interpolated_density_4champ_jb, interpolated_density_4champ_tie, measurement_data, time_array, figsize=(12,5), time_indexer = 'None'):
    """
    plots all data
    interpolated_density_4champ_jb: that returned by interpolate_density for jb2008 model
    interpolated_density_4champ_tie: that returned by interpolate_density for tie-gcm model
    measurement_data: accelerometer data, take it from the df_hourly[header[-3]]
    time_array: the datetimes, take it from the df_hourly
    
    returns fig, ax if needed
    """
            
    assert (time_indexer in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'None'])
    time_dictionary_model = {'None': 0, 'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
                       'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    
    if time_indexer == 'None':
        time_start = 0
        time_end = len(time_array)
    else:
        month1 = time_dictionary_model[time_indexer]
        day1 = 1
        time_start = 0
        for i in range(len(time_array)):
            if time_array[i].month == month1 and time_start == 0:
                time_start = i
            elif time_array[i].month == month1+1:
                time_end = i
                break
        if month1 == 12:
            time_end = len(time_array)
    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize = figsize)
    ax.plot(time_array[time_start:time_end],interpolated_density_4champ_jb[time_start:time_end], '-.', label='jb2008 model') #, '.', ms = 5
    ax.plot(time_array[time_start:time_end],interpolated_density_4champ_tie[time_start:time_end], '--', label='tie-gcm model') #, '.', ms = 5
    ax.plot(time_array[time_start:time_end],measurement_data[time_start:time_end], '-*', label='acceleromater-derived data')
    ax.set_ylabel("density value [kg/m^3]", size=14, fontfamily='georgia')
    ax.set_xlabel("time [YYYY-MM-DD]", size=14, fontfamily='georgia')
    plt.title('comparison of density values for CHAMP trajectory', fontfamily='georgia', size=16)
    plt.legend()
    plt.plot()
    return fig, ax


def daily_average_calculator2(df_hourly, header, interpolated_density_4champ_jb, interpolated_density_4champ_tie, time_indexer = 'None'):
    """
    calculates the hourly averages for model interpol. and the measurement data
    takes into account missing dates
    
    df_hourly, header: you know the drill by now
    interpolated_density_4champ_jb: that returned by interpolate_density for jb2008 model
    interpolated_density_4champ_tie: that returned by interpolate_density for tie-gcm model

    time_indexer: if you want a specific month, input its name as a string
    
    returns: avg_jb,avg_tie,avg_acc, time_to_plot
    
    averaged jb2008
    averaged tie-g..
    averaged measurement
    datetimes corresponding
    
    """
    
    assert (time_indexer in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'None'])
    time_dictionary_model = {'None': 0, 'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
                       'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                       
    avg_jb = []
    avg_tie = []
    avg_acc = []
    time_to_plot = []
    temp_jb = 0
    temp_tie = 0
    temp_acc = 0
    counter = 0
    if time_indexer == 'None':
        time_start = 0
        time_end = len(df_hourly[header[0]])
        month1 = df_hourly[header[0]][0].month
        day1 = df_hourly[header[0]][0].day
        time_to_plot.append(df_hourly[header[0]][time_start])
    else:
        month1 = time_dictionary_model[time_indexer]
        day1 = 1
        time_start = 0
        for i in range(len(df_hourly[header[0]])):
            if df_hourly[header[0]][i].month == month1 and time_start == 0:
                time_start = i
            elif df_hourly[header[0]][i].month == month1+1:
                time_end = i
                break
        time_to_plot.append(df_hourly[header[0]][time_start])
    if month1 == 12:
        time_end = len(df_hourly[header[0]])
    for index in range(time_start,time_end):
        if df_hourly[header[0]][index].month == month1 and df_hourly[header[0]][index].day == day1:
            temp_jb = temp_jb + interpolated_density_4champ_jb[index]
            temp_tie = temp_tie + interpolated_density_4champ_tie[index]
            temp_acc = temp_acc + df_hourly[header[-3]][index]
            counter = counter + 1
        else:
            avg_jb.append(temp_jb/counter)
            avg_tie.append(temp_tie/counter)
            avg_acc.append(temp_acc/counter)
            time_to_plot.append(time_to_plot[-1] + datetime.timedelta(days=1))
            temp_jb = 0
            temp_tie = 0
            temp_acc = 0
            counter = 0
            month1 = df_hourly[header[0]][index].month
            day1 = df_hourly[header[0]][index].day
            temp_jb = temp_jb + interpolated_density_4champ_jb[index]
            temp_tie = temp_tie + interpolated_density_4champ_tie[index]
            temp_acc = temp_acc + df_hourly[header[-3]][index]
            counter = counter + 1            
    avg_jb.append(temp_jb/counter)
    avg_tie.append(temp_tie/counter)
    avg_acc.append(temp_acc/counter)
    #time_to_plot.append(time_to_plot[-1] + datetime.timedelta(days=1))
    temp_jb = 0
    temp_tie = 0
    temp_acc = 0
    counter = 0
    return(avg_jb,avg_tie,avg_acc, time_to_plot)

def read_ascii_file(filename,index=-1,starttime=datetime.datetime(1950,1,1),endtime=datetime.datetime(2050,1,1)):    
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
    assert isinstance(starttime,datetime.datetime), "check starttime input"
    assert isinstance(endtime,datetime.datetime), "check endtime input"  
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
            data["time"].append(datetime.datetime(year,1,1) + datetime.timedelta(days = doy-1,hours=hour,minutes=minute))
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

def compare_symh_density(champ_data, header, symh_data, figsize=(12,5)):
    fig, ax = plt.subplots(nrows=2, ncols = 1, figsize = figsize)
    ax[0].plot(champ_data[header[0]],champ_data[header[-3]], '.', label='CHAMP density') #, '.', ms = 5
    ax[1].plot(symh_data['time'],symh_data['symh'], '--', label='SYM-H data') #, '.', ms = 5
    ax[0].set_ylabel("density value [kg/m^3]", size=14, fontfamily='georgia')
    ax[1].set_ylabel("SYM-H value [nT]", size=14, fontfamily='georgia')
    ax[1].set_xlabel("time [YYYY-MM-DD]", size=14, fontfamily='georgia')
    ax[0].set_title('comparison of SYM-H to density value', fontfamily='georgia', size=16)
    plt.legend()
    plt.plot()
    return fig, ax

                    
if __name__ == '__main__':
    
    route = r"C:/Users/Niki/Documents/GIT/swsss2023/CHAMP_New" #/   Champ_dens_2002
    files = os.listdir(route)
    files = [route + r"/" + x for x in files]

    CHAMP_data, header_var = champ_reader(files)
    CHAMP_data_hourly = create_hourly_data(CHAMP_data, header_var)

    dir_density_Jb2008 = r"C:\Users\Niki\Documents\GIT\swsss2023\day_03\JB2008\2002_JB2008_density.mat"
    dir_density_tiegcm = r"C:\Users\Niki\Documents\GIT\swsss2023\day_03\TIEGCM\2002_TIEGCM_density.mat"
    loaded_data = h5py.File(dir_density_tiegcm)
    
    interpol_champ_jb = interpolate_density(dir_density_Jb2008, CHAMP_data_hourly, header_var)
    interpol_champ_tie = interpolate_density(dir_density_tiegcm, CHAMP_data_hourly, header_var, model='tiegcm')
    #check = convert_to_datetime(CHAMP_data_hourly, header_var)
        
    plot_density_data(interpol_champ_jb,(CHAMP_data_hourly[header_var[0]]))
    plot_density_data(interpol_champ_tie,(CHAMP_data_hourly[header_var[0]]),name_modifier=2)
    plot_density_data(CHAMP_data_hourly[header_var[-3]][:],(CHAMP_data_hourly[header_var[0]]), name_modifier=3)
    
    compare_density_data(interpol_champ_jb, interpol_champ_tie, (CHAMP_data_hourly[header_var[0]]))
    
    plot_all_data(interpol_champ_jb, interpol_champ_tie,CHAMP_data_hourly[header_var[-3]][:],(CHAMP_data_hourly[header_var[0]]), time_indexer='July') # , time_indexer='January'
    
    daily_jb, daily_tie, daily_acc, plot_time = daily_average_calculator2(CHAMP_data_hourly, header_var, interpol_champ_jb, interpol_champ_tie) #, time_indexer="December"
    
    plot_all_data(daily_jb, daily_tie, daily_acc, plot_time)
    
    omni_data = r"C:/Users/Niki/Documents/GIT/swsss2023/day_05/2002_omni_data.lst"
    
    symh_data = pd.DataFrame(read_ascii_file(omni_data))
    symh_hourly = create_hourly_data((symh_data), ['time', 'symh']) 

    compare_symh_density(CHAMP_data_hourly, header_var, symh_hourly)

    

