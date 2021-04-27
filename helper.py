import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

c = 3e8
m = 9.11e-31
e = 1.6e-19
epsilon0 = 8.85e-12

UNITS = {'m':1,'cm':0.01,'mm':0.001,'um':10**(-6)}

def normalize(value,unit,plasma_density): # plasma_density is in 10^16 /cm^3
    omega_p = np.sqrt(plasma_density * 1e22 * e * e/ epsilon0 / m)

    value *= UNITS[unit]
    return value / (c / omega_p)

def unnormalize(value,unit,plasma_density): # plasma_density is in 10^16 /cm^3
    omega_p = np.sqrt(plasma_density * 1e22 * e * e/ epsilon0 / m)
    value *= (c / omega_p)
    return value / UNITS[unit]

# This function set the longitudinal varying plasma density for each species in QPAD input file
def set_plasma_density(density,s,path = '..'):

    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    nspecies = inputDeck['simulation']['nspecies']
    for i in range(nspecies):
        inputDeck['species'][i]['piecewise_fs'] = list(density)
        inputDeck['species'][i]['piecewise_s'] = list(s)

    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

# This function plots the plasma density of the first species in QPAD input file

def plot_plasma_density(path = '..',save=False):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    fs = inputDeck['species'][0]['piecewise_fs']
    s = inputDeck['species'][0]['piecewise_s']
    plt.plot(s,fs)
    plt.xlabel('z')
    plt.ylabel(r'$n(z)/n_0$')
    plt.title('Plasma density profile')
    if save:
        plt.savefig('plasma_density_profile_in_qpinput.png')
    plt.show()
