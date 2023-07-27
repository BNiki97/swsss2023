"""TEC plotter code."""

__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

from wam_ipe_plotter import saving_tec_plot
from sys import argv
for i in range(1,len(argv)):
    saving_tec_plot(argv[i])