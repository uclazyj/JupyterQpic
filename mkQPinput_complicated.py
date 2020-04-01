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

# Physical constants

c = 3e8
m = 9.11e-31
e = 1.6e-19
epsilon0 = 8.85e-12
        
def plotSourceProfile():
    rb = 2
    Delta = 0.2
    epsilon = 0.01
    peak = rb**2/((rb+Delta)**2 - rb**2)
    r = [0,rb,rb+epsilon,rb+Delta,rb+Delta+epsilon,3*rb]
    source = [-1,-1,peak,peak,0,0]
    plt.plot(r,source)
    plt.xlabel('$r$')
    plt.ylabel('$-(\\rho - J_z)$')
    #ylim([-1,5])
    plt.yticks(np.arange(-1, 6, 1))
    plt.title('$r_b = 2, \Delta = 0.1$') 
    plt.show()

def makeInput(inputDeckTemplateName,units,
              plasmaDensityProfile,plasmaDataFile,zDataFile,
                 indx,indz,n0,dt,nbeams,time,ndump2D,
                 boxXlength,boxYlength,boxZlength,
                 z_driver,
                 sigma_x_driver,sigma_y_driver,sigma_z_driver,
                 sigma_vx_driver,sigma_vy_driver,
                 gammaE_driver,energySpread_driver,
                 peak_density_driver,
                 z_witness,
                 sigma_x_witness,sigma_y_witness,sigma_z_witness,
                 sigma_vx_witness,sigma_vy_witness,
                 gammaE_witness,energySpread_witness,
                 peak_density_witness): 
    
    if(plasmaDensityProfile == 'piecewise'):
        piecewise_density = np.loadtxt(fname = plasmaDataFile, delimiter=",")
        piecewise_z = np.loadtxt(fname = zDataFile, delimiter=",")
    
    # If the unit system is SI, then the peak density is the total charge, and all the quantities should be converted into 
    # normalized units
    if(units == 'SI'):
        # Convert everything into SI units first
        boxXlength = boxXlength / 1e6; boxYlength = boxYlength / 1e6; boxZlength = boxZlength / 1e6;
        z_driver = z_driver / 1e6; z_witness = z_witness/1e6;
        sigma_x_driver = sigma_x_driver/1e6; sigma_y_driver = sigma_y_driver/1e6; sigma_z_driver = sigma_z_driver/1e6;
        sigma_x_witness = sigma_x_witness/1e6; sigma_y_witness = sigma_y_witness/1e6; sigma_z_witness = sigma_z_witness/1e6;
        if(plasmaDensityProfile == 'piecewise'):
            piecewise_z = piecewise_z/1e6
            
        # These peak_densities are actually Q. Convert the units from nC to C
        Q_driver = peak_density_driver/1e9;Q_witness = peak_density_witness/1e9;
        
        # Calculate the peak densities (normalized to plasma density)
        peak_density_driver = Q_driver/e/sqrt((2*pi)**3) / sigma_x_driver / sigma_y_driver / sigma_z_driver / (n0 * 1e22)
        peak_density_witness = Q_witness/e/sqrt((2*pi)**3) / sigma_x_witness / sigma_y_witness / sigma_z_witness / (n0 * 1e22)
        
        # Convert all the quantities from SI units to normalized units (except the peak densities, which are ready to use)
        wp = sqrt(n0 * 1e22 * e * e / epsilon0 / m)
        kp = 1/(c / wp);
        
        boxXlength = boxXlength * kp; boxYlength = boxYlength * kp; boxZlength = boxZlength * kp;
        z_driver = z_driver * kp; z_witness = z_witness * kp;
        sigma_x_driver = sigma_x_driver * kp; sigma_y_driver = sigma_y_driver * kp; sigma_z_driver = sigma_z_driver * kp;
        sigma_x_witness = sigma_x_witness * kp; sigma_y_witness = sigma_y_witness * kp; sigma_z_witness = sigma_z_witness * kp;
        if(plasmaDensityProfile == 'piecewise'):
            piecewise_z = piecewise_z * kp
    
    # Get the file object from the template QuickPIC input file

    with open(inputDeckTemplateName) as ftemplate:
        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)

    # Modify the parameters in the QuickPIC input file

    inputDeck['simulation']['indx'] = indx
    inputDeck['simulation']['indy'] = indx
    inputDeck['simulation']['indz'] = indz
    inputDeck['simulation']['n0'] = n0 * 1e16
    
    inputDeck['simulation']['dt'] = dt
    inputDeck['simulation']['nbeams'] = nbeams
    inputDeck['simulation']['time'] = time
    
    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
    inputDeck['simulation']['box']['y'][0] = - boxYlength / 2
    inputDeck['simulation']['box']['y'][1] = boxYlength / 2
    inputDeck['simulation']['box']['z'][1] = boxZlength
    
    inputDeck['beam'][0]['gamma'] = gammaE_driver
    inputDeck['beam'][0]['peak_density'] = peak_density_driver
    
    inputDeck['beam'][0]['center'][2] = z_driver
    inputDeck['beam'][0]['sigma'] = [sigma_x_driver, sigma_y_driver, sigma_z_driver]
    inputDeck['beam'][0]['sigma_v'] = [sigma_vx_driver,sigma_vy_driver,gammaE_driver * energySpread_driver/100]
    
    inputDeck['beam'][1]['gamma'] = gammaE_witness
    inputDeck['beam'][1]['peak_density'] = peak_density_witness
    
    inputDeck['beam'][1]['center'][2] = z_witness
    inputDeck['beam'][1]['sigma'] = [sigma_x_witness,sigma_y_witness,sigma_z_witness]
    inputDeck['beam'][1]['sigma_v'] = [sigma_vx_witness,sigma_vy_witness,gammaE_witness * energySpread_witness/100]
    
    inputDeck['species'][0]['longitudinal_profile'] = plasmaDensityProfile
    if(plasmaDensityProfile == 'piecewise'):
        inputDeck['species'][0]['piecewise_density'] = list(piecewise_density)
        inputDeck['species'][0]['piecewise_s'] = list(piecewise_z)
    
    ################# Diagnostic #################
    
    xzSlicePosition = 2 ** (indx - 1) + 1
    yzSlicePosition = 2 ** (indx - 1) + 1
    xySlicePosition = 2 ** (indx - 2) * 3
    # 2D xz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['field']['diag'][1]['slice'][0][1] = xzSlicePosition
    # 2D yz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['field']['diag'][1]['slice'][1][1] = yzSlicePosition
    # 2D xy slice position
    inputDeck['beam'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['field']['diag'][1]['slice'][2][1] = xySlicePosition
    # 2D dumping frequency
    inputDeck['beam'][0]['diag'][1]['ndump'] = ndump2D
    inputDeck['beam'][0]['diag'][2]['ndump'] = ndump2D
    inputDeck['beam'][1]['diag'][1]['ndump'] = ndump2D
    inputDeck['beam'][1]['diag'][2]['ndump'] = ndump2D
    inputDeck['species'][0]['diag'][1]['ndump'] = ndump2D
    inputDeck['field']['diag'][1]['ndump'] = ndump2D
    # Save the changes to 'qpinput.json'

    with open('qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    
def makeWidgetsForInput():        
    style = {'description_width': '350px'}
    layout = Layout(width='55%')
    
    inputDeckTemplateNameW = widgets.Text(value='qpinput_template.json', description='Template Input File:',style=style,layout=layout)

    unitsW = widgets.Dropdown(options=['Normalized', 'SI'],value='SI', description='Units:',style=style,layout=layout)
    plasmaDensityProfileW = widgets.Dropdown(options=['uniform', 'piecewise'],value='uniform', description='Longitudinal Plasma Density Profile:',style=style,layout=layout)
    
    plasmaDataFileW = widgets.Text(value='plasma.txt', description='Plasma Data File:',style=style,layout=layout)
    zDataFileW = widgets.Text(value='z.txt', description='z Data File (Normalized/$\mu$m):',style=style,layout=layout)
    
    indxW = widgets.IntText(value=9, description='indx (indy):', style=style, layout=layout)
    indzW = widgets.IntText(value=9, description='indz:', style=style, layout=layout)


    n0W = widgets.FloatText(value=3.5, description='$n_0\;(10^{16}/cm^3)$:', style=style, layout=layout)

    dtW = widgets.IntText(value=10, description='dt:', style=style, layout=layout)
    #nbeamsW = widgets.IntText(value=2, description='nbeams:', style=style, layout=layout)
    nbeamsW = widgets.IntSlider(value=2,min=1,max=2,step=1, description='number of beams:',style=style, layout=layout)
    timeW = widgets.FloatText(value=10540, description='time:', style=style, layout=layout)
    ndump2DW = widgets.IntText(value=20, description='dump the data every ? time steps:', style=style, layout=layout)
    
    boxXlengthW = widgets.FloatText(value=500, description='boxXlength (Normalized/$\mu$m):', style=style, layout=layout)
    boxYlengthW = widgets.FloatText(value=500, description='boxYlength (Normalized/$\mu$m):', style=style, layout=layout)
    boxZlengthW = widgets.FloatText(value=200, description='boxZlength (Normalized/$\mu$m):', style=style, layout=layout)
    
    # Driving beam

    z_driverW = widgets.FloatText(value=30, description='driver z position (Normalized/$\mu$m):', style=style, layout=layout)

    sigma_x_driverW = widgets.FloatText(value=10.25, description='$\sigma_x$(Normalized/$\mu$m) (driver):', style=style, layout=layout)
    sigma_y_driverW = widgets.FloatText(value=10.25, description='$\sigma_y$(Normalized/$\mu$m) (driver):', style=style, layout=layout)
    sigma_z_driverW = widgets.FloatText(value=6.4, description='$\sigma_z$(Normalized/$\mu$m) (driver):', style=style, layout=layout)
    
    sigma_vx_driverW = widgets.FloatText(value=0, description='$\sigma_{px}$ (driver):', style=style, layout=layout)
    sigma_vy_driverW = widgets.FloatText(value=0, description='$\sigma_{py}$ (driver):', style=style, layout=layout)
    
    gammaE_driverW = widgets.FloatText(value=20000, description='$\gamma$ (driver):', style=style, layout=layout)    
    energySpread_driverW = widgets.FloatText(value=0.25, description='$\Delta \gamma /\gamma$ (%) (driver):', style=style, layout=layout)
    peak_density_driverW = widgets.FloatText(value=1.6, description='$n_{peak}$ (Normalized) or $Q_{total}$ (nC) (driver):', style=style, layout=layout)

    # Witness beam

    z_witnessW = widgets.FloatText(value=180, description='witness z position (Normalized/$\mu$m):', style=style, layout=layout)

    sigma_x_witnessW = widgets.FloatText(value=0.9468, description='$\sigma_x$(Normalized/$\mu$m) (witness):', style=style, layout=layout)
    sigma_y_witnessW = widgets.FloatText(value=0.9468, description='$\sigma_y$(Normalized/$\mu$m) (witness):', style=style, layout=layout)
    sigma_z_witnessW = widgets.FloatText(value=5, description='$\sigma_z$(Normalized/$\mu$m) (witness):', style=style, layout=layout)
    
    sigma_vx_witnessW = widgets.FloatText(value=3.327, description='$\sigma_{px}$ (witness):', style=style, layout=layout)
    sigma_vy_witnessW = widgets.FloatText(value=3.327, description='$\sigma_{py}$ (witness):', style=style, layout=layout)
    
    gammaE_witnessW = widgets.FloatText(value=20000, description='$\gamma$ (witness):', style=style, layout=layout)    
    energySpread_witnessW = widgets.FloatText(value=0.25, description='$\Delta \gamma /\gamma$ (%) (witness):', style=style, layout=layout)
    peak_density_witnessW = widgets.FloatText(value=0.5, description='$n_{peak}$ (Normalized) or $Q_{total}$ (nC) (witness):', style=style, layout=layout)
    
    interact_calc(makeInput,inputDeckTemplateName = inputDeckTemplateNameW,units = unitsW,
                  plasmaDensityProfile = plasmaDensityProfileW,plasmaDataFile = plasmaDataFileW,zDataFile = zDataFileW,
                  indx = indxW,indz=indzW,n0 = n0W,dt=dtW,nbeams=nbeamsW,time = timeW,ndump2D = ndump2DW,
                  boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
                  z_driver = z_driverW,
                  sigma_x_driver = sigma_x_driverW,sigma_y_driver = sigma_y_driverW,sigma_z_driver = sigma_z_driverW,  
                  sigma_vx_driver = sigma_vx_driverW,sigma_vy_driver = sigma_vy_driverW,
                  gammaE_driver = gammaE_driverW,energySpread_driver = energySpread_driverW,
                  peak_density_driver = peak_density_driverW,
                  z_witness = z_witnessW,
                  sigma_x_witness = sigma_x_witnessW,sigma_y_witness = sigma_y_witnessW,sigma_z_witness = sigma_z_witnessW,  
                  sigma_vx_witness = sigma_vx_witnessW,sigma_vy_witness = sigma_vy_witnessW,
                  gammaE_witness = gammaE_witnessW,energySpread_witness = energySpread_witnessW,
                  peak_density_witness = peak_density_witnessW);


