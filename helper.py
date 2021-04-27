import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

c = 3e8
m = 9.11e-31
e = 1.6e-19
epsilon0 = 8.85e-12

UNITS = {'m':1,'cm':0.01,'mm':0.001,'um':10**(-6)}

def normalize(value,unit,plasma_density): # plasma_density is in cm^-3
    omega_p = np.sqrt(plasma_density * 1e6 * e * e/ epsilon0 / m)

    value *= UNITS[unit]
    return value / (c / omega_p)

def to_phys_unit(value,unit,plasma_density): # plasma_density is in cm^-3
    omega_p = np.sqrt(plasma_density * 1e6 * e * e/ epsilon0 / m)
    value *= (c / omega_p)
    return value / UNITS[unit]

# This function set the longitudinal varying plasma density for each species in QPAD input file
def set_plasma_density(idx,density,s,path = '..'):

    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)

    inputDeck['species'][idx]['piecewise_fs'] = list(density)
    inputDeck['species'][idx]['piecewise_s'] = list(s)

    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

# This function plots the plasma density of the first species in QPAD input file

def plot_plasma_density(save=False,path = '..'):
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

# This function sets the beam parameters sigma and sigma_v so that the beam is matched to the uniform plasma
# idx: The beam idx in input file (0 may correspond to the drive beam, 1 may correspond to the witness beam)
# epsilon_n is in normalized unit
def set_matched_beam(idx,epsilon_n,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    gamma = inputDeck['beam'][idx]['gamma']
    epsilon = epsilon_n / gamma
    beta_m = np.sqrt(2 * gamma)
    sigma_m = np.sqrt(beta_m * epsilon)
    sigma_p = epsilon_n / sigma_m
    inputDeck['beam'][idx]['sigma'][0] = sigma_m
    inputDeck['beam'][idx]['sigma'][1] = sigma_m
    inputDeck['beam'][idx]['sigma_v'][0] = sigma_p
    inputDeck['beam'][idx]['sigma_v'][1] = sigma_p
    
    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    return sigma_m

# This function uses the input parameter N (the total number of electrons in the beam) and the 
# sigma_x, sigma_y, sigma_z specified in the input file to calculate the beam peak density (normalized to n0), and set it in the input file.
def set_beam_peak_density(idx,N,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    n0 = inputDeck['simulation']['n0']
    sigma_x, sigma_y, sigma_z = inputDeck['beam'][idx]['sigma']
    sigma_x = to_phys_unit(sigma_x,'cm',n0)
    sigma_y = to_phys_unit(sigma_y,'cm',n0)
    sigma_z = to_phys_unit(sigma_z,'cm',n0)
    n_peak = N / (2*np.pi)**(3/2) / (sigma_x * sigma_y * sigma_z) / n0
    inputDeck['beam'][idx]['peak_density'] = n_peak
    
    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    
    return n_peak

# This function sets all the ndump in the input file
def set_ndump(ndump,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    nbeam = inputDeck['simulation']['nbeams']
    nspecies = inputDeck['simulation']['nspecies']
    nneutrals = inputDeck['simulation']['nneutrals']
    # set ndump for the beams
    for i in range(nbeam):
        ndiag = len(inputDeck['beam'][i]['diag'])
        for j in range(ndiag):
            inputDeck['beam'][i]['diag'][j]['ndump'] = ndump
    # set ndump for the species
    for i in range(nspecies):
        ndiag = len(inputDeck['species'][i]['diag'])
        for j in range(ndiag):
            inputDeck['species'][i]['diag'][j]['ndump'] = ndump
    # set ndump for the neutrals
    for i in range(nneutrals):
        ndiag = len(inputDeck['neutrals'][i]['diag'])
        for j in range(ndiag):
            inputDeck['neutrals'][i]['diag'][j]['ndump'] = ndump
    # set ndump for the fields
    inputDeck['field']['diag'][0]['ndump'] = ndump
        
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
def get_n0(path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    return inputDeck['simulation']['n0']