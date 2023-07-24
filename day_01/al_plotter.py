"""A code for AL plotting"""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

import datetime
from swmfpy.web import get_omni_data
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def get_AL_data(start_time):
    
    """
    This function downloads data from the OMNI database for a set time,
    plots the AL index for one whole day,
    then returns the time and AL index as lists.
    Input variable:
        start date
        give it in the format of datetime(YEAR,MONTH,DAY)
        the function will then add another day for the end time
    Plotted data:
        x axis: date (HH:MM)
        y axis: AL index
        time resolution: one day
    Output:
        the function will return the time and AL index as lists for further use
    """

    end_time = start_time + datetime.timedelta(days=1)
    data = get_omni_data(start_time, end_time)
    time_list = list(data['times'])
    AL_list = list(data['al'])
    fig, ax = plt.subplots()
    ax.plot(time_list,AL_list)
    ax.set(title="AL Index Data for One Day's Time\nStart Time: {0}".format(start_time))
    ax.set_xlabel("date (HH:MM)")
    ax.set_ylabel("AL index")
    ax.xaxis.set_major_locator(plt.MaxNLocator(9))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.show()
    return(time_list, AL_list)

time, al = get_AL_data(datetime.datetime(1997,1,26))
