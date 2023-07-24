"""A 3D plot script for spherical coordinates"""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

# x = r*sin(fi)*cos(theta)
# y = r*sin(fi)*sin(theta)
# z = r*cos(fi)

from math import sin, cos, pi
import matplotlib.pyplot as plt
import numpy as np

def spherical_to_cartesian(radius, azimuth, zenith):
    """This function converts spherical coordinates to cartesian.
    The function only takes one value for each variable.
    Each value must be a float or an integer.
    Input variables:
        1st r: the radius;
        2nd theta: the azimuth angle;
        3rd phi: the zenith angle;
    Output variables:
        1st x: the x coordinate
        2nd y: the y coordinate
        3rd z: the z coordinate
    ATTENTION:
        The function ONLY TAKES ONE VALUE FOR EACH VARIABLE.
        If you want to convert an array of variables, use list_conversion_helper.
    """
    assert (isinstance(radius,float) or isinstance(radius,int)),'check r value'
    assert (isinstance(azimuth,float) or isinstance(azimuth, int)), 'check azimuth value'
    assert (isinstance(zenith,float) or isinstance(zenith, int)), 'check zenith value'
    x = radius*sin(zenith)*cos(azimuth)
    y = radius*sin(zenith)*sin(azimuth)
    z = radius*cos(zenith)  
    return(x, y, z)

def list_conversion_helper(radius, azimuth, zenith):
    """This function helps convert using spherical_to_cartesian when you have more than one value.
    Input:
        lists of radius, azimuth and zenith
    Output:
        x, y, z
    """
    coord_dict = {'r': radius, #np.linspace(0,1),
                  'theta': azimuth, #np.linspace(0,2*pi),
                  'phi': zenith} #np.linspace(0,2*pi)}    
    x = [spherical_to_cartesian(coord_dict['r'][item], coord_dict['theta'][item], coord_dict['phi'][item])[0] for item in range(len(coord_dict['r']))]
    y = [spherical_to_cartesian(coord_dict['r'][item], coord_dict['theta'][item], coord_dict['phi'][item])[1] for item in range(len(coord_dict['r']))]
    z = [spherical_to_cartesian(coord_dict['r'][item], coord_dict['theta'][item], coord_dict['phi'][item])[2] for item in range(len(coord_dict['r']))]
    return(x,y,z)
    
if __name__ == '__main__':  # main code block
    
    # CHECKING IF THE CONVERSION WORKS
    assert np.allclose([spherical_to_cartesian(1, 0, 0)], [0,0,1]), 'conversion not working as expected'
    assert np.allclose([spherical_to_cartesian(1, pi, pi)], [0,0,-1]), 'conversion not working as expected'
    assert np.allclose([spherical_to_cartesian(1, 2*pi, 2*pi)], [0,0,1]), 'conversion not working as expected'
    assert np.allclose([spherical_to_cartesian(1, -pi, -2*pi)], [0,0,1]), 'conversion not working as expected'
    assert np.allclose([spherical_to_cartesian(1, -2*pi, pi)], [0,0,-1]), 'conversion not working as expected'

    # PLOTTING A TRAJECTORY
    x,y,z = list_conversion_helper(np.linspace(0,1),np.linspace(0,2*pi),np.linspace(0,2*pi))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x,y,z)
    
    #r = np.linspace(0,1)
    #theta = np.linspace(0,2*pi)
    #phi = np.linspace(0,2*pi)
    #newvec = np.transpose(np.mat([r, theta, phi]))

"""
    # RANDOM CONVERSIONS
    cartesian=spherical_to_cartesian(0.5, 30, 90)
    print(cartesian)
    cartesian = spherical_to_cartesian(1, 0, 0)
    print(cartesian)
    cartesian = spherical_to_cartesian(1, -pi, -2*pi)
    print(cartesian)
    cartesian = spherical_to_cartesian(1, -2*pi, pi)
    print(cartesian)
"""    







