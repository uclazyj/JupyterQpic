import os
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, FloatSlider, HBox, VBox, interactive_output

import json
from collections import OrderedDict
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
from math import *
interact_calc=interact_manual.options(manual_name="Make New Input!")

def analyzeWitnessRawData(zVisualizeCenter,halfThickness,en,e,a,b,sig,energy,energySpread):
    
    with open('qpinput.json') as f:
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
    
    nbeams = inputDeck['simulation']['nbeams']
    if(nbeams == 1):
        return
    dt = inputDeck['simulation']['dt']
    time = inputDeck['simulation']['time']
    NtimeSteps = int(time/dt)
    ndump2D = inputDeck['beam'][1]['diag'][1]['ndump']
    
    zWitnessCenter = inputDeck['beam'][1]['center'][2]
    gamma = inputDeck['beam'][1]['gamma']
    
    profile = inputDeck['beam'][1]['profile']
    if(profile == 0):
        sigmaz= inputDeck['beam'][1]['sigma'][2]
        sigma = inputDeck['beam'][1]['sigma'][0]
        sigma_p = inputDeck['beam'][1]['sigma_v'][0]
        emitn_i = sigma * sigma_p
        alpha_i = 0
        beta_i = sigma ** 2 / (emitn_i / gamma)
        energySpread = inputDeck['beam'][1]['sigma_v'][2] / gamma
    elif(profile == 2):
        sigmaz= inputDeck['beam'][1]['sigmaz']  
        emitn_i = inputDeck['beam'][1]['emittance'][0]
        alpha_i = inputDeck['beam'][1]['alpha'][0]
        beta_i = inputDeck['beam'][1]['beta'][0]
        energySpread = inputDeck['beam'][1]['sigma_vz'] / gamma
        
    zVisualizeMax = zWitnessCenter + zVisualizeCenter * sigmaz + halfThickness * sigmaz
    zVisualizeMin = zWitnessCenter + zVisualizeCenter * sigmaz - halfThickness * sigmaz

    emitn_x_z = np.array([])
    emit_x_z = np.array([])
    emitn_y_z = np.array([])
    emit_y_z = np.array([])
    gammaE_z = np.array([])
    energySpread_z = np.array([])
    alpha_x_z = np.array([])
    alpha_y_z = np.array([])
    beta_x_z = np.array([])
    beta_y_z = np.array([])
    sigma_x_z = np.array([])
    sigma_y_z = np.array([])

    timeSteps = range(1,NtimeSteps,ndump2D)
    
    # Calculate the theoretical emittance growth
    gamma_i = (1 + alpha_i ** 2) / beta_i
    beta_m = sqrt(2 * gamma) # normalized unit
    A = (gamma_i * beta_m + beta_i / beta_m )/2
    
    z_plasma = np.array(timeSteps) * dt
    phi =  z_plasma / beta_m
    emitEvolUniformPlasmaTheory = emitn_i * np.sqrt(A**2 - (A**2-1) * np.exp(-(energySpread * phi)**2))
    
    cwd = os.getcwd()
    os.chdir(cwd+'/Beam0002/Raw')
    
    for timeStep in timeSteps:

        timeStep = str(timeStep).zfill(8)
        filename = 'raw_' + timeStep + '.h5'
        f=h5py.File(filename,'r')
        
        dataset_x3 = f['/x3'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        z = dataset_x3[...] # type(data) outputs numpy.ndarray
        inVisualizationRange = (z > zVisualizeMin) & (z < zVisualizeMax)
        z = z[inVisualizationRange]
        
        dataset_p1 = f['/p1'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        px = dataset_p1[...] # type(data) outputs numpy.ndarray
        px = px[inVisualizationRange] # extract the part within the data visualization range
        px = px - px.mean()
        dataset_p2 = f['/p2'] 
        py = dataset_p2[...]
        py = py[inVisualizationRange]
        py = py - py.mean()
        dataset_p3 = f['/p3'] 
        gammaE = dataset_p3[...] 
        gammaE = gammaE[inVisualizationRange]
        gammaE_bar = gammaE.mean()
        gammaE_z = np.append(gammaE_z,gammaE_bar)
        sigma_gammaE = np.std(gammaE)
        energySpread_z = np.append(energySpread_z,sigma_gammaE / gammaE_bar)
        
        xprime = px / gammaE
        yprime = py / gammaE

        dataset_x1 = f['/x1'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        x = dataset_x1[...] # type(data) outputs numpy.ndarray
        x = x[inVisualizationRange]
        x = x - x.mean()
        sigma_x = np.std(x)
        sigma_x_z = np.append(sigma_x_z,sigma_x)

        dataset_x2 = f['/x2'] 
        y = dataset_x2[...]
        y = y[inVisualizationRange]
        y = y - y.mean()
        sigma_y = np.std(y)
        sigma_y_z = np.append(sigma_y_z,sigma_y)

        emitn_x = np.sqrt(sigma_x ** 2 * np.std(px) ** 2 - (x * px).mean() ** 2)
        emitn_x_z = np.append(emitn_x_z,emitn_x)
        emit_x = np.sqrt(sigma_x ** 2 * np.std(xprime) ** 2 - (x * xprime).mean() ** 2)
        emit_x_z = np.append(emit_x_z,emit_x)
        emitn_y = np.sqrt(sigma_y ** 2 * np.std(py) ** 2 - (y * py).mean() ** 2)
        emitn_y_z = np.append(emitn_y_z,emitn_y)
        emit_y = np.sqrt(sigma_y ** 2 * np.std(yprime) ** 2 - (y * yprime).mean() ** 2)
        emit_y_z = np.append(emit_y_z,emit_y)
        
        alpha_x = -(x * xprime).mean() / emit_x
        alpha_x_z = np.append(alpha_x_z,alpha_x)
        alpha_y = -(y * yprime).mean() / emit_y
        alpha_y_z = np.append(alpha_y_z,alpha_y)
        
        beta_x = sigma_x ** 2 / emit_x
        beta_x_z = np.append(beta_x_z,beta_x)
        beta_y = sigma_y ** 2 / emit_y
        beta_y_z = np.append(beta_y_z,beta_y)
    
    # Plot the beam quantities according to the user's need
    if(en):
        fig, ax = plt.subplots()
        plt.plot(timeSteps, emitn_x_z,label='x')
        plt.plot(timeSteps, emitn_y_z,label='y')
        plt.plot(timeSteps, emitEvolUniformPlasmaTheory,label='Theory')
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\epsilon_n \;(c/\omega_p)$')
        plt.legend(loc='lower right')
        plt.show()
        fig.savefig('emittance.png')
    if(e):
        plt.plot(timeSteps, emit_x_z,label='x')
        plt.plot(timeSteps, emit_y_z,label='y')
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\epsilon \;(c/\omega_p)$')
        plt.legend(loc='lower right')
        plt.show()
    if(a):
        plt.plot(timeSteps, alpha_x_z,label='x')
        plt.plot(timeSteps, alpha_y_z,label='y')
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\\alpha$')
        plt.legend(loc='lower right')
        plt.show()
    if(b):
        plt.plot(timeSteps, beta_x_z,label='x')
        plt.plot(timeSteps, beta_y_z,label='y')
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\\beta \;(c/\omega_p)$')
        plt.legend(loc='lower right')
        plt.show()
    
    if(sig):
        plt.plot(timeSteps, sigma_x_z,label='x')
        plt.plot(timeSteps, sigma_y_z,label='y')
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\sigma \;(c/\omega_p)$')
        plt.legend(loc='lower right')
        plt.show()
    
    if(energy):
        plt.plot(timeSteps, gammaE_z)
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\gamma $')
        plt.show()
    if(energySpread):
        plt.plot(timeSteps, energySpread_z)
        plt.xlabel('$z\;(c/\omega_p)$')
        plt.ylabel('$\Delta \gamma /\gamma$ (%)')
        plt.show()
    
    os.chdir(cwd)
    return


def chooseWhatToSee():
    interact(analyzeWitnessRawData,
             zVisualizeCenter = widgets.IntSlider(value=0,min=-3,max=3,step=1, description='center of data visualizatino region'),
             halfThickness = widgets.FloatText(value=1, description='half thickness of data visualization region'),
             en = widgets.Checkbox(value=True,description='$\epsilon_n$'),
             e = widgets.Checkbox(value=False,description='$\epsilon$'),
             a = widgets.Checkbox(value=False,description='$\\alpha$'),
             b = widgets.Checkbox(value=False,description='$\\beta$'),
             sig = widgets.Checkbox(value=False,description='$\sigma$'),
             energy = widgets.Checkbox(value=False,description='$\gamma$'),
             energySpread = widgets.Checkbox(value=False,description='$\Delta \gamma /\gamma$ (%)'),
            );
    return
