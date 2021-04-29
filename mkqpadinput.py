### Users can set input parameters in this file. 
### This file will call a bunch of functions in helper.py (helper.py must be in the same folder as this file) 
### to calculate and set the parameters in qpinput.json conveniently

### Before using this script, you should set n0 (plasma density) and gamma,sigma_z for each beam correctly in qpinput.json
import numpy as np
import matplotlib.pyplot as plt
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
ndump = 15
######## END SET PARAMETERS SECTION ########

def sigmoid(z,a):
    return 1 / (1 + np.exp(-z/a))

beta_mi = normalize(5,'cm',n0)
beta_mf = 312.4
a = 3000
z = np.linspace(-3 * a,9 * a, 1201)
sig = sigmoid(z,a)
beta_m = (1-sig) * (beta_mi - beta_mf) + beta_mf
alpha_m = sig * (1 - sig) * (beta_mi - beta_mf) / 2 / a
# print('alpha_m[0] = ',alpha_m[0])
# print('alpha_m[-1] = ',alpha_m[-1])
n = (beta_mf / beta_m) ** 2
s = z - z[0]
set_plasma_density(s,n,'species',0,'..')
set_plasma_density(s,n,'species',1,'..')

### DO NOT CHANGE THE CODE BELOW ###

### Set parameters for drive beam ###
idx = 0
set_matched_beam(idx = idx,epsilon_n = epsilon_n_drive,uniform = True,name = 'species',i = 0,path = path)
set_beam_peak_density(idx,N_drive,path)
### Set parameters for witness beam ###
idx = 1
set_matched_beam(idx = idx,epsilon_n = epsilon_n_witness,uniform = False,name = 'species',i = 0,path = path)
set_beam_peak_density(idx,N_witness,path)
### set ndump ###
set_ndump(ndump,path)
get_density_profile(name = 'species', idx = 0, plot = True, save=False, path = '..')
