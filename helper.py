import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import h5py
import pyVisQP

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

# Perform a linear regression, and return the predicted values for x
# Input x, y are both 1D array
def linear_regression(x,y):
    if len(x) != len(y):
        print('The length of the input must match!')
        return
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    reg.fit(x,y)
    pred = reg.predict(x)
    return pred.reshape(-1)

def get_one_item(keylist,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        a = json.load(f,object_pairs_hook=OrderedDict)
    for i in keylist:
        a = a[i]
    return a

# delete the species and beams that are not used in the input file
def clean_up(path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    nspecies = inputDeck['simulation']['nspecies']
    nbeams = inputDeck['simulation']['nbeams']

    while len(inputDeck['species']) > nspecies:
        inputDeck['species'].pop()
    while len(inputDeck['beam']) > nbeams:
        inputDeck['beam'].pop()
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

# fill up the species and beams in the input file (append copies of the first beam until there are nbeam beams. Also append copies of the first species until there are nspecies)
def fill(path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    nspecies = inputDeck['simulation']['nspecies']
    nbeams = inputDeck['simulation']['nbeams']

    while len(inputDeck['species']) < nspecies:
        inputDeck['species'].append(inputDeck['species'][0])
    while len(inputDeck['beam']) < nbeams:
        inputDeck['beam'].append(inputDeck['beam'][0])
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

# This function set the longitudinal varying plasma density for each species in QPAD input file
def set_plasma_density(s,density,name = 'species',idx = 0,path = '..'):
    if len(s) != len(density):
        print('The length of s and fs do not match!')
        return
    if s[0] != 0:
        print('The s array should start from 0!')
        return

    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    
    inputDeck[name][idx]['piecewise_s'] = list(s)
    inputDeck[name][idx]['piecewise_fs'] = list(density)
    inputDeck[name][idx]['profile'][1] = 'piecewise-linear'
    inputDeck['simulation']['time'] = s[-1]

    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

def set_uniform_plasma(name = 'species',idx = 0,path = '..'):

    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    if 'piecewise_s' in inputDeck[name][idx]:
        del inputDeck[name][idx]['piecewise_s']
    if 'piecewise_fs' in inputDeck[name][idx]:
        del inputDeck[name][idx]['piecewise_fs']

    inputDeck[name][idx]['profile'][1] = 'uniform'

    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
# This function plots the plasma density of the first species in QPAD input file

def get_density_profile(name = 'species', idx = 0, plot = False, save=False, path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    s = inputDeck[name][idx]['piecewise_s']
    fs = inputDeck[name][idx]['piecewise_fs']
    if plot:
        plt.plot(s,fs)
        plt.xlabel('z')
        plt.ylabel(r'$n(z)/n_0$')
        plt.title('Plasma density profile')
        if save:
            plt.savefig('plasma_density_profile_in_qpinput.png')
        plt.show()
    return (s,fs)

def get_numbers_in_filenames(path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    ndump = inputDeck['beam'][0]['diag'][-1]['ndump']
    time = inputDeck['simulation']['time']
    dt = inputDeck['simulation']['dt']
    last_file_number = int(time // ( dt * ndump) * ndump )
    timeSteps = np.array(range(0,last_file_number+ndump,ndump))
    s = timeSteps * dt
    return timeSteps,s

# This function sets the beam parameters sigma and sigma_v so that the beam is matched to the uniform plasma (if uniform = True) or the plasma entrance (if uniform = False)
# idx: The beam idx in input file (0 may correspond to the drive beam, 1 may correspond to the witness beam)
# epsilon_n is in normalized unit
# local_density is the local plasma density normalized to n0
# name and i determines the profile of which species or neutral that the beam is matching to
def set_matched_beam(idx,epsilon_n,name = 'species',i = 0,uniform = True,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    entrance_density = 1.0
    if uniform == False: # Then match the beam to the plasma entrance
        s,fs = get_density_profile(name,i,False,False,path)
        entrance_density = fs[0]

    gamma = inputDeck['beam'][idx]['gamma']
    epsilon = epsilon_n / gamma
    beta_m = np.sqrt(2 * gamma)
    sigma_m = np.sqrt(beta_m * epsilon)
    sigma_m = sigma_m / np.sqrt(np.sqrt(entrance_density)) # sigma_m^4 * n is a constant (assuming geometric emittance is a constant)
    sigma_p = epsilon_n / sigma_m
    inputDeck['beam'][idx]['gauss_sigma'][0:2] = [sigma_m,sigma_m]
    inputDeck['beam'][idx]['uth'][0:2] = [sigma_p,sigma_p]
    
    ## Write the modified file object into a jason file
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    return sigma_m

# This function uses the input parameter N (the total number of electrons in the beam) and the 
# sigma_x, sigma_y, sigma_z specified in the input file to calculate the beam peak density (normalized to n0), and set it in the input file.
def set_beam_peak_density(idx,N,path = '..',QPAD = True):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    n0 = inputDeck['simulation']['n0']
    profile = inputDeck['beam'][idx]['profile']
    if (QPAD and profile == 3) or (not QPAD and profile == 2):
        beta_x,beta_y = inputDeck['beam'][idx]['beta']
        epsilon_n_x,epsilon_n_y = inputDeck['beam'][idx]['emittance']
        gamma = inputDeck['beam'][idx]['gamma']
        sigma_x = np.sqrt(beta_x * epsilon_n_x / gamma)
        sigma_y = np.sqrt(beta_y * epsilon_n_y / gamma)
        sigma_z = inputDeck['beam'][idx]['sigmaz']
    else:
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
    nneutrals = 0
    if 'nneutrals' in inputDeck['simulation']:
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
    ndiag = len(inputDeck['field']['diag'])
    for j in range(ndiag):
        inputDeck['field']['diag'][j]['ndump'] = ndump
        
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
def get_n0(path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    return inputDeck['simulation']['n0']   

# From the plasma density in the input file, return the matched parameters that match to the r/2 focusing force
# i is the beam index
def get_matched_beam_parameters(i = 1,name = 'species',idx = 0,path = '..',QPAD = True):
    parameters = {}
    s, n = get_density_profile(name, idx,False,False,path)
    parameters['s'] = s
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    profile = inputDeck['beam'][i]['profile']
    gamma = inputDeck['beam'][i]['gamma']

    epsilon_n = inputDeck['beam'][i]['gauss_sigma'][0] * inputDeck['beam'][i]['uth'][0]
    
    beta_m0 = np.sqrt(2*gamma)
    sigma_m0 = np.sqrt(beta_m0 * epsilon_n / gamma)
    beta_m = beta_m0 / np.sqrt(n)
    sigma_m = sigma_m0 / np.sqrt(np.sqrt(n))
    parameters['sigma_m'] = sigma_m
    parameters['beta_m'] = beta_m
    parameters['alpha_m'] = -1/2 * pyVisQP.NDiff1D(s,beta_m)
    return parameters

# Used for QPAD profile 3 or QuickPIC profile 2. Calculate the initial spotsize from the input file
def get_init_spot_size(i,path = '..',QPAD = True):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    beta = inputDeck['beam'][i]['beta'][0]
    epsilon_n = inputDeck['beam'][i]['emittance'][0]
    gamma = inputDeck['beam'][i]['gamma']
    sigma = np.sqrt(beta * epsilon_n / gamma)
    return sigma

def change_spotsize_fix_charge_and_emittance(i,times,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    inputDeck['beam'][i]['sigma'][0] *= times
    inputDeck['beam'][i]['sigma'][1] *= times
    inputDeck['beam'][i]['sigma_v'][0] /= times
    inputDeck['beam'][i]['sigma_v'][1] /= times
    inputDeck['beam'][i]['peak_density'] /= (times ** 2)
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
def get_lineout_idx(Min,Max,lineout_pos,n_grid_points):
    if lineout_pos < Min or lineout_pos > Max:
        print('Lineout position is out of range:[',Min,',',Max,']!')
        return 0
    return int((n_grid_points-1) / (Max - Min) * (lineout_pos - Min) + 0.5)


def get_lineout(filename,direction,lineout_pos,code = 'QPAD'):
    with h5py.File(filename, 'r') as h5file:
        dset_name = list(h5file.keys())[1] # dset_name = 'charge_slice_xz'
        data = np.array(h5file[dset_name])
        n_grid_points_xi, n_grid_points_x = data.shape
        x_range = np.array(h5file['AXIS']['AXIS1'])
        xi_range = np.array(h5file['AXIS']['AXIS2'])

        if direction == 'transverse':
            lineout_idx_xi = get_lineout_idx(xi_range[0],xi_range[1],lineout_pos,n_grid_points_xi)
            lineout = data[lineout_idx_xi,:]
            if code == 'QuickPIC':
                lineout = lineout[1:] # get rid of the empty data in the first column

            x = np.linspace(x_range[0],x_range[1],num = len(lineout))
            return x,lineout
        elif direction == 'longitudinal':
            lineout_idx_x = get_lineout_idx(x_range[0],x_range[1],lineout_pos,n_grid_points_x)
            lineout = data[:,lineout_idx_x]

            xi = np.linspace(xi_range[0],xi_range[1],num = len(lineout))
            return xi,lineout

def get_raw_beam_data(filename):
    
    f=h5py.File(filename,'r')
    raw_data = {}
    
    dataset_x1 = f['/x1'] 
    x = dataset_x1[...]
    raw_data['x'] = x

    dataset_p1 = f['/p1'] 
    px = dataset_p1[...]
    raw_data['px'] = px

    dataset_x2 = f['/x2'] 
    y = dataset_x2[...]
    raw_data['y'] = y

    dataset_p2 = f['/p2'] 
    py = dataset_p2[...]
    raw_data['py'] = py

    dataset_x3 = f['/x3'] 
    z = dataset_x3[...] 
    raw_data['z'] = z

    dataset_p3 = f['/p3'] 
    pz = dataset_p3[...] 
    raw_data['pz'] = pz

    dataset_q = f['/q']
    q = dataset_q[...]
    raw_data['q'] = q
    
    return raw_data
    
def select_lineout_range(x,lineout,x_min,x_max):
    if x_min < x[0] or x_max > x[-1] or x_min > x_max:
        print('Invalid lineout range!')
        return x,lineout
    x_min_idx = get_lineout_idx(x[0],x[-1],x_min,len(x)) 
    x_max_idx = get_lineout_idx(x[0],x[-1],x_max,len(x))
    return x[x_min_idx:x_max_idx+1],lineout[x_min_idx:x_max_idx+1]


# The following functions are for QuickPIC

def propagate_alpha_beta(alpha_i,beta_i,z):
    gamma_i = (1 + alpha_i ** 2) / beta_i
    beta = beta_i - 2 * z * alpha_i + z ** 2 * gamma_i
    alpha = alpha_i - z * gamma_i
    return (alpha, beta)

# s is the position of the beam's vacuum waist inside the plasma (at the plasma entrance, the position is 0)
# (relative to the beam's vacuum waist, the plasma entrance is at -s from the beam's waist)
def set_Twiss_at_entrance(i,beta_star,s,path = '..',QPAD = True): # i is the beam index
    alpha, beta = propagate_alpha_beta(0,beta_star,-s)
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)

    inputDeck['beam'][i]['profile'] = 3 if QPAD else 2
    inputDeck['beam'][i]['alpha'] = [alpha,alpha]
    inputDeck['beam'][i]['beta'] = [beta,beta]

    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
def get_Twiss_in_vacuum(alpha_i,beta_i,z): # alpha_i and beta_i are numbers, s is a numpy array
    gamma_i = (1 + alpha_i ** 2) / beta_i
    alpha = alpha_i - z * gamma_i
    beta = beta_i - 2 * z * alpha_i + z ** 2 * gamma_i
    return alpha,beta
    

def set_one_item(type_name,idx,parameter_name,value,path = '..'):
    with open(path + '/qpinput.json') as f:
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    inputDeck[type_name][idx][parameter_name] = value
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    

# This function sets all the slice index
def set_slice_idx(center = True,slice_idx=0,path = '..'):
    with open(path + '/qpinput.json') as f: # This is the old jason input file
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    if center:
        indx = inputDeck['simulation']['indx']
        slice_idx = 2 ** indx // 2 + 1
    nbeam = inputDeck['simulation']['nbeams']
    nspecies = inputDeck['simulation']['nspecies']
    nneutrals = 0
    if 'nneutrals' in inputDeck['simulation']:  
        nneutrals = inputDeck['simulation']['nneutrals']
    # set slice idx for the beams
    for i in range(nbeam):
        ndiag = len(inputDeck['beam'][i]['diag'])
        for j in range(ndiag):
            if 'slice' in inputDeck['beam'][i]['diag'][j]:
                nslices = len(inputDeck['beam'][i]['diag'][j]['slice'])
                for k in range(nslices):
                    if inputDeck['beam'][i]['diag'][j]['slice'][k][0] in {"xz","yz"}:
                        inputDeck['beam'][i]['diag'][j]['slice'][k][1] = slice_idx
    
    # set slice idx for the species
    for i in range(nspecies):
        ndiag = len(inputDeck['species'][i]['diag'])
        for j in range(ndiag):
            if 'slice' in inputDeck['species'][i]['diag'][j]:
                nslices = len(inputDeck['species'][i]['diag'][j]['slice'])
                for k in range(nslices):
                    if inputDeck['species'][i]['diag'][j]['slice'][k][0] in {"xz","yz"}:
                        inputDeck['species'][i]['diag'][j]['slice'][k][1] = slice_idx

    # set slice idx for the neutrals
    for i in range(nneutrals):
        ndiag = len(inputDeck['neutrals'][i]['diag'])
        for j in range(ndiag):
            if 'slice' in inputDeck['neutrals'][i]['diag'][j]:
                nslices = len(inputDeck['neutrals'][i]['diag'][j]['slice'])
                for k in range(nslices):
                    if inputDeck['neutrals'][i]['diag'][j]['slice'][k][0] in {"xz","yz"}:
                        inputDeck['neutrals'][i]['diag'][j]['slice'][k][1] = slice_idx
    # set slice idx for the fields
    ndiag = len(inputDeck['field']['diag'])
    for j in range(ndiag):
        if 'slice' in inputDeck['field']['diag'][j]:
            nslices = len(inputDeck['field']['diag'][j]['slice'])
            for k in range(nslices):
                if inputDeck['field']['diag'][j]['slice'][k][0] in {"xz","yz"}:
                    inputDeck['field']['diag'][j]['slice'][k][1] = slice_idx
    with open(path + '/qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)

        
