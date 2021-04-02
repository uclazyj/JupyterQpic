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
