### Users can set input parameters in this file. 
### This file will call a bunch of functions in helper.py (helper.py must be in the same folder as this file) 
### to calculate and set the parameters in qpinput.json conveniently

### Before using this script, you should set n0 (plasma density) and gamma,sigma_z for each beam correctly in qpinput.json

from importlib import reload
import helper
from helper import *
helper = reload(helper)

### Get plasma density from qpinput.json
path = '..' # relative path to qpinput.json
n0 = get_n0(path)

######## SET PARAMETERS SECTION ##########
N_drive = 3*10**10 # number of electrons 
N_witness = 10**10
epsilon_n_drive = normalize(1,'mm',n0)
epsilon_n_witness = normalize(0.1,'um',n0)
ndump = 111
######## END SET PARAMETERS SECTION ########




### DO NOT CHANGE THE CODE BELOW ###

### Set parameters for drive beam ###
idx = 0
set_matched_beam(idx,epsilon_n_drive,path)
set_beam_peak_density(idx,N_drive,path)
### Set parameters for witness beam ###
idx = 1
set_matched_beam(idx,epsilon_n_witness,path)
set_beam_peak_density(idx,N_witness,path)
### set ndump ###
set_ndump(ndump,path)
