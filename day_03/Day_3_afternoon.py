#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Creating a random numpy array
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

#%%
"""
TODO: Writing and reading numpy file
"""
np.save('test_np_save.npy',data_arr)

data_arr_loaded = np.load('test_np_save.npy')

print(np.equal(data_arr,data_arr_loaded))
print(data_arr == data_arr_loaded)

#%%
"""
TODO: Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_np_save.npz',data_arr,data_arr2)

# Load the numpy zip file
npzipfile = np.load('test_np_save.npz')
print(npzipfile)
print(sorted(npzipfile.files))

# Verify that the loaded data matches the initial data

#%%
"""
Error and exception
"""
# Exception handling, can be use with assertion as well
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(np.equal(data_arr,npzipfile).all())
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("The codes in try returned an error.")
    print(np.equal(data_arr,npzipfile['arr_0']).all())
    
#%%
"""
TODO: Error solving 1
"""
# What is wrong with the following line? 
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(np.equal(data_arr,data_arr2))
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("Those two arrays have different shapes, cannot be compared.")


#%%
"""
TODO: Error solving 2
"""
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(np.equal(data_arr2,npzipfile['data_arr2']))
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("That is the wrong key.")
    print(np.equal(data_arr2,npzipfile['arr_1']))

#%%
"""
TODO: Error solving 3
"""
# What is wrong with the following line? 
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(numpy.equal(data_arr2,npzipfile['arr_1']))
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("numpy is input as np")
    print(np.equal(data_arr2,npzipfile['arr_1']))



#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = r"C:\Users\Niki\Documents\GIT\swsss2023\day_03\JB2008\2002_JB2008_density.mat"

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the 
# discretization grid of the density data in 3D space. We will be using 
# np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8759,5, dtype = int)

# For the dataset that we will be working with today, you will need to reshape 
# them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                               nofAlt_JB2008,8760), order='F') # Fortran-like index order

#%%
"""
TODO: Plot the atmospheric density for 400 KM for the first time index in
      time_array_JB2008 (time_array_JB2008[0]).
"""

import matplotlib.pyplot as plt

# Look for data that correspond to an altitude of 400 KM
alt = 400
hi = np.where(altitudes_JB2008==alt)
        
dens_to_plot = JB2008_dens_reshaped[:,:,hi,0]
dens_to_plot_reshape = np.reshape(dens_to_plot,(nofLst_JB2008,nofLat_JB2008)).transpose()
#.squeeze()

fig,ax = plt.subplots()
ax.contourf(localSolarTimes_JB2008,latitudes_JB2008,dens_to_plot_reshape)
    
    #r = np.linspace(0,1)
    #theta = np.linspace(0,2*pi)
    #phi = np.linspace(0,2*pi)
    #newvec = np.transpose(np.mat([r, theta, phi]))



#%%
"""
TODO: Plot the atmospheric density for 300 KM for all time indexes in
      time_array_JB2008
"""
alt = 300
hi = np.where(altitudes_JB2008==alt)
fig, axs = plt.subplots(5,figsize=(15,12))
for i in range(len(time_array_JB2008)):
    dens_to_plot = JB2008_dens_reshaped[:,:,hi,i]
    dens_to_plot_reshape = np.reshape(dens_to_plot,(nofLst_JB2008,nofLat_JB2008)).transpose()
    heh=axs[i].contourf(localSolarTimes_JB2008,latitudes_JB2008,dens_to_plot_reshape)
    cbar = fig.colorbar(heh)
plt.show()

    

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]
print('The dimension of the data are as followed(local solar time,latitude,altitude):', dens_data_feb1.shape)
data_to_plot_alt = [np.mean(dens_data_feb1[:,:,x]) for x in range(nofAlt_JB2008)]
fig,ax = plt.subplots()  
ax.set_yscale('log')
ax.plot(altitudes_JB2008[:],data_to_plot_alt[:])
plt.grid()
plt.show()



#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density 
field at 310km

"""
# Import required packages
import h5py
loaded_data = h5py.File(r"C:\Users\Niki\Documents\GIT\swsss2023\day_03\TIEGCM\2002_TIEGCM_density.mat")

# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within dataset:',list(loaded_data.keys()))

tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

# We will be using the same time index as before.
time_array_tiegcm = time_array_JB2008
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')


#%%
"""
TODO: Plot the atmospheric density for 310 KM for all time indexes in
      time_array_tiegcm
"""

alt = 310
hi = np.where(altitudes_tiegcm==alt)
fig, axs = plt.subplots(5,figsize=(15,12))
for i in range(len(time_array_tiegcm)):
    dens_to_plot = tiegcm_dens_reshaped[:,:,hi,i]
    dens_to_plot_reshape = np.reshape(dens_to_plot,(nofLst_tiegcm,nofLat_tiegcm)).transpose()
    heh=axs[i].contourf(localSolarTimes_tiegcm,latitudes_tiegcm,dens_to_plot_reshape)
    cbar = fig.colorbar(heh)
plt.show()


#%%
"""
Assignment 1.5

Can you plot the mean density for each altitude at February 1st, 2002 for both 
models (JB2008 and TIE-GCM) on the same plot?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]
dens_data_feb2 = tiegcm_dens_reshaped[:,:,:,time_index]
print('The dimension of the data are as followed(local solar time,latitude,altitude):', dens_data_feb1.shape)
data_to_plot_alt = [np.mean(dens_data_feb1[:,:,x]) for x in range(nofAlt_JB2008)]
data_to_plot_alt2 = [np.mean(dens_data_feb2[:,:,x]) for x in range(nofAlt_tiegcm)]
fig,ax = plt.subplots()  
ax.set_yscale('log')
ax.plot(altitudes_JB2008[:],data_to_plot_alt[:], label='JB2008')
ax.plot(altitudes_tiegcm[:],data_to_plot_alt2[:],'-', label='tiegcm')
plt.grid()
plt.legend()
plt.show()

print(JB2008_dens_reshaped.shape,tiegcm_dens_reshaped.shape)

#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10, 3)
y = np.exp(-x/3.0)

interp_func_1D = interpolate.interp1d(x,y)

xnew = np.arange(0,9,0.1)
ynew = interp_func_1D(xnew)

interp_func_1D_cub = interpolate.interp1d(x,y,kind='cubic')
ycubic = interp_func_1D_cub(xnew)

interp_func_1d_quadr = interpolate.interp1d(x,y,kind='quadratic')
yquad = interp_func_1d_quadr(xnew)

plt.plot(xnew,ynew)
plt.plot(x,y)


fig=plt.subplots(1, figsize=(10,6))
plt.plot(x,y,'o', xnew, ynew, '*', xnew, ycubic, '--', xnew, yquad, '--', linewidth=2)
plt.legend(['initial points', 'interpolated linear', 'interpolated cubic', 'interpolated quadratic'], fontsize = 16)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title('1D interpolation', fontsize=18)
plt.grid()
plt.tick_params(axis='both',which='major', labelsize=16)
plt.show()

#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)
interpolated_function_1 = RegularGridInterpolator((x,y,z), sample_data)

pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print("using interpolation method: ", interpolated_function_1(pts))
print("from true function: ", function_1(pts[:,0], pts[:,1], pts[:, 2]))


#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization 
grid.

Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on 
February 1st, 2002, with the discretized grid used for the JB2008 
((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""
time_index = 31*24
sample_data = tiegcm_dens_reshaped[:,:,:,time_index]

x = localSolarTimes_tiegcm 
y = latitudes_tiegcm
z = altitudes_tiegcm # (400)

interpolated_function__tiegcm = RegularGridInterpolator((x,y,z), sample_data, bounds_error=False, fill_value=None)

#ptsx, ptsy, ptsz = np.meshgrid(localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008, indexing='ij', sparse=True)
pts = np.zeros((nofLst_JB2008,nofLat_JB2008))
for i in range(nofLst_JB2008):
    for j in range(nofLat_JB2008):
        pts_index = np.array([localSolarTimes_JB2008[i], latitudes_JB2008[j], 400])
        pts[i,j] = interpolated_function__tiegcm((pts_index))

"""
# this works as well!
xx = localSolarTimes_JB2008
yy = latitudes_JB2008
zz = 400
X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')
interp_to_plot = interpolated_function__tiegcm((X, Y, Z)).reshape(nofLst_JB2008,nofLat_JB2008)
"""

alt = 400
hi = np.where(altitudes_JB2008==alt)
dens_to_plot = JB2008_dens_reshaped[:,:,hi,time_index]
dens_to_plot_reshape = np.reshape(dens_to_plot,(nofLst_JB2008,nofLat_JB2008))
fig, axs = plt.subplots(2,figsize=(15,12))
plt.title(label='400 km data for February 2002', loc="left")
heh=axs[0].contourf(localSolarTimes_JB2008,latitudes_JB2008,pts.T) # or interp_to_plot.T
axs[0].set_title('interpolated tiegcm data')
cbar = fig.colorbar(heh)
heh2=axs[1].contourf(localSolarTimes_JB2008,latitudes_JB2008,dens_to_plot_reshape.T)
axs[1].set_title('original jb2008 data')
cbar = fig.colorbar(heh2)
plt.show()

#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this 
difference in a contour plot.
"""

I_can_make_a_difference = np.zeros((nofLst_JB2008,nofLat_JB2008))
for i in range(nofLst_JB2008):
    for j in range(nofLat_JB2008):
        I_can_make_a_difference[i,j] = pts[i,j] - dens_to_plot_reshape[i,j]

fig, axs = plt.subplots(1,figsize=(15,6))
plt.title(label='difference between original jb2008 data and interpolated tiegcm data')
heh=axs.contourf(localSolarTimes_JB2008,latitudes_JB2008,I_can_make_a_difference.T) # or interp_to_plot.T
cbar = fig.colorbar(heh)
plt.show()


#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in 
terms of absolute percentage difference/error (APE). Let's plot the APE 
for this scenario.

APE = abs(tiegcm_dens-JB2008_dens)/tiegcm_dens
"""

I_can_make_more_difference = np.zeros((nofLst_JB2008,nofLat_JB2008))
for i in range(nofLst_JB2008):
    for j in range(nofLat_JB2008):
        I_can_make_more_difference[i,j] = abs(I_can_make_a_difference[i,j]) / pts[i,j]

fig, axs = plt.subplots(1,figsize=(15,6))
plt.title(label='absolute percentage difference between original jb2008 data and interpolated tiegcm data')
heh=axs.contourf(localSolarTimes_JB2008,latitudes_JB2008,I_can_make_more_difference.T) # or interp_to_plot.T
cbar = fig.colorbar(heh)
plt.show()

