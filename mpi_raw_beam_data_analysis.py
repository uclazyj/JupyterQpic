from mpi4py import MPI
from time import time

import json
from collections import OrderedDict
from importlib import reload
import helper
helper = reload(helper)
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import pyVisQP
import os
dirname = '..'

t0 = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

half_thickness = 5
zVisualizeCenter = 0

# The following is the default setting to analyze raw beam data for all output files
QPAD = True

timeSteps,_ = get_numbers_in_filenames()
# timeSteps = timeSteps[:3]
# print(timeSteps)
nbeams = get_one_item(['simulation','nbeams'])
beam_number = nbeams # Usually, the witness beam is the last beam in the input file

### Plot the emittance evolution for multiple slices in witness beam

xi_s = [-1.0,-0.5,0.0,0.5,1.0]
half_thickness_slice = 0.1

xi = xi_s[rank]
parameters_xi = pyVisQP.analyze_raw_beam_data_remove_outliers(timeSteps = timeSteps, beam_number = beam_number, \
                                                  zVisualizeCenter = xi, half_thickness = half_thickness_slice)
parameters_xi_s = comm.gather(parameters_xi,root = 0)

if rank == 0:
    pyVisQP.save_beam_analysis(beam_number,xi_s,parameters_xi_s,half_thickness_slice)