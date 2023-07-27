"""TEC plotter code."""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import matplotlib.pyplot as plt
import datetime
import netCDF4 as nc

def plot_tec(dataset, figsize=(12,6)):
    """
    FUNCTION TO PLOT TEC DATA
    
    INPUT:
        dataset: NetCDF file containing TEC data
        figsize: figure size, default: (12,6)
    
    OUTPUT:
        the function will plot the data, but will also return fig and ax varibles
        
    The function extracts the tec, latitude and longitude keys, then plots
    tec as a colormesh, the function of longitude-latitude
    """

    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize = figsize)
    tec_data = dataset['tec'][:]
    unit = dataset['tec'].units
    latitude_data = dataset['lat'][:]
    lat_unit = dataset['lat'].units
    longitude_data = dataset['lon'][:]
    lon_unit = dataset['lon'].units
    start_date = datetime.datetime.strptime(dataset.start_date, "%Y%m%d_%H%M%S")
    to_color=ax.pcolormesh(longitude_data,latitude_data,tec_data, cmap='plasma')
    ax.set_ylabel("latitude [{0}]".format(lat_unit), size=14, fontfamily='georgia')
    ax.set_xlabel("longitude [{0}]".format(lon_unit), size=14, fontfamily='georgia')
    plt.title('TEC data for start date of {0}'.format(start_date), fontfamily='georgia', size=16)
    plt.colorbar(to_color, label='value [{0}]'.format(unit))
    plt.plot()
    
    return fig, ax

def plot_wam_ipe(dataset, key = 'tec', figsize=(12,6)):
    """
    FUNCTION TO PLOT WAM IPE DATA
    
    INPUT:
        dataset: NetCDF file containing data
        key: the variable to plot, will default to TEC if not given
        figsize: figure size, default: (12,6)
    
    OUTPUT:
        the function will plot the data, but will also return fig and ax varibles
        
    The function extracts the key, latitude and longitude, then plots
    the key variable as a colormesh, the function of longitude-latitude
    """
    
    assert (key in dataset.variables), "this key doesn't exist"
    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize = figsize)
    to_plot_data = dataset[key][:]
    unit = dataset[key].units
    latitude_data = dataset['lat'][:]
    lat_unit = dataset['lat'].units
    longitude_data = dataset['lon'][:]
    lon_unit = dataset['lon'].units
    start_date = datetime.datetime.strptime(dataset.init_date, "%Y%m%d_%H%M%S")
    if key == 'tec':
        to_color=ax.pcolormesh(longitude_data,latitude_data,to_plot_data, cmap='plasma', vmin=0, vmax=130)
    else:
        to_color=ax.pcolormesh(longitude_data,latitude_data,to_plot_data, cmap='plasma')
    ax.set_ylabel("latitude [{0}]".format(lat_unit), size=14, fontfamily='georgia')
    ax.set_xlabel("longitude [{0}]".format(lon_unit), size=14, fontfamily='georgia')
    plt.title('{0} data for date of {1}'.format(key.upper(),start_date), fontfamily='georgia', size=16)
    plt.colorbar(to_color, label='value [{0}]'.format(unit))
    plt.plot()
    
    return fig, ax

def saving_tec_plot(infilename, key = 'tec'):
    """
    FUNCTION TO SAVE WAM IPE DATA
    
    INPUT:
        infilename: the name and route to the dataset to open
        will be the name and route by which the plot will be saved
        key = the value to plot, defaults to 'tec'
    
    OUTPUT:
        the function doesn't return anything, just saves the plot
        
    """
    fig, _ = plot_wam_ipe(nc.Dataset(infilename),key)
    outfilename = infilename + r".png"
    fig.savefig(outfilename, format='png')
    plt.close()

if __name__ == '__main__':
    
    dataset = nc.Dataset(r"C:/Users/Niki/Documents/GIT/swsss2023/day_02/wfs.t06z.ipe05.20230726_091000.nc")
    fig, ax = plot_tec(dataset)

    fig, ax = plot_wam_ipe(dataset,'tec')
    
    name_to_file = r"C:/Users/Niki/Documents/GIT/swsss2023/day_02/wfs.t06z.ipe05.20230726_091000.nc"
    saving_tec_plot(name_to_file)

