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

def makeInput(inputDeckTemplateName,units,
                 indx,indz,n0,time,dt,nbeams,
                 boxXlength,boxYlength,boxZlength,
                 z_driver,
                 sigma_x_driver,sigma_y_driver,sigma_z_driver,
                 gammaE_driver,
                 peak_density_driver,
                 z_witness,
                 sigma_r_witness,epsilon_n_witness,sigma_z_witness,
                 gammaE_witness,energy_spread_witness,
                 peak_density_witness,
                 ): 
    
    # Convert all the quantities from SI units to normalized units (except the peak densities, which are ready to use)
    wp = sqrt(n0 * 1e22 * e * e / epsilon0 / m)
    kp = 1/(c / wp);
   
    # Get the file object from the template QuickPIC input file

    with open(inputDeckTemplateName) as ftemplate:
        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)

    # Modify the parameters in the QuickPIC input file

    inputDeck['simulation']['indx'] = indx
    inputDeck['simulation']['indy'] = indx
    inputDeck['simulation']['indz'] = indz
    inputDeck['simulation']['n0'] = n0 * 1e16
    inputDeck['simulation']['time'] = time
    inputDeck['simulation']['dt'] = dt
    inputDeck['simulation']['nbeams'] = nbeams
    
    if(units == 'Experimental'):
        boxXlength = boxXlength / 1e6 * kp
        boxZlength = boxZlength / 1e6 * kp
        
        z_driver = z_driver / 1e6 * kp 
        
        sigma_x_driver /= 1e6
        sigma_y_driver /= 1e6
        sigma_z_driver /= 1e6
  
        # The peak_densities are actually Q 
        peak_density_driver = peak_density_driver/1e9/e/sqrt((2*pi)**3) / sigma_x_driver / sigma_y_driver / sigma_z_driver / (n0 * 1e22)
        
        sigma_x_driver *= kp
        sigma_y_driver *= kp
        sigma_z_driver *= kp
        
        z_witness = z_witness / 1e6 * kp 
        
        sigma_r_witness /= 1e6
        sigma_z_witness /= 1e6
        
        # The peak_densities are actually Q 
        peak_density_witness = peak_density_witness/1e9/e/sqrt((2*pi)**3) / sigma_r_witness / sigma_r_witness / sigma_z_witness / (n0 * 1e22)
        
        sigma_r_witness *= kp
        sigma_z_witness *= kp
        
        epsilon_n_witness = epsilon_n_witness / 1e6 * kp 
    
    
    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
    inputDeck['simulation']['box']['y'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['y'][1] = boxXlength / 2
    inputDeck['simulation']['box']['z'][1] = boxZlength
    
    inputDeck['beam'][0]['gamma'] = gammaE_driver
    inputDeck['beam'][0]['peak_density'] = peak_density_driver
    
    inputDeck['beam'][0]['center'][2] = z_driver
    inputDeck['beam'][0]['sigma'] = [sigma_x_driver, sigma_y_driver, sigma_z_driver]
    
    inputDeck['beam'][1]['gamma'] = gammaE_witness
    inputDeck['beam'][1]['peak_density'] = peak_density_witness
    
    inputDeck['beam'][1]['center'][2] = z_witness
    inputDeck['beam'][1]['sigma'] = [sigma_r_witness, sigma_r_witness, sigma_z_witness]
    inputDeck['beam'][1]['sigma_v'] = [epsilon_n_witness/sigma_r_witness, epsilon_n_witness/sigma_r_witness, energy_spread_witness * gammaE_witness]

    
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

    # Save the changes to 'qpinput.json'

    with open('qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    
def makeWidgetsForInput():        
    style = {'description_width': '350px'}
    layout = Layout(width='55%')
    
    inputDeckTemplateNameW = widgets.Text(value='qpinput_template.json', description='Template Input File:',style=style,layout=layout)
    
    unitsW = widgets.Dropdown(options=['Normalized', 'Experimental'],value='Experimental', description='Units:',style=style,layout=layout)
    
    indxW = widgets.IntText(value=9, description='indx (indy):', style=style, layout=layout)
    indzW = widgets.IntText(value=9, description='indz:', style=style, layout=layout)


    n0W = widgets.FloatText(value=3.5, description='$n_0\;(10^{16}/cm^3)$:', style=style, layout=layout)
    timeW = widgets.IntText(value=2, description='time:', style=style, layout=layout)
    dtW = widgets.IntText(value=1, description='dt:', style=style, layout=layout)
    nbeamsW = widgets.IntText(value=2, description='nbeams:', style=style, layout=layout)
    
    boxXlengthW = widgets.FloatText(value=500, description='boxXlength (Normalized/$\mu m$):', style=style, layout=layout)
    boxYlengthW = widgets.FloatText(value=500, description='boxYlength (Normalized/$\mu m$):', style=style, layout=layout)
    boxZlengthW = widgets.FloatText(value=200, description='boxZlength (Normalized/$\mu m$):', style=style, layout=layout)
    
    # Driving beam

    z_driverW = widgets.FloatText(value=30, description='driver z position (Normalized/$\mu m$):', style=style, layout=layout)

    sigma_x_driverW = widgets.FloatText(value=10.25, description='$\sigma_x$ (Normalized/$\mu m$):', style=style, layout=layout)
    sigma_y_driverW = widgets.FloatText(value=10.25, description='$\sigma_y$ (Normalized/$\mu m$):', style=style, layout=layout)
    sigma_z_driverW = widgets.FloatText(value=6.4, description='$\sigma_z$ (Normalized/$\mu m$):', style=style, layout=layout)
    
    gammaE_driverW = widgets.FloatText(value=20000, description='$\gamma$:', style=style, layout=layout)    
    peak_density_driverW = widgets.FloatText(value=1.6, description='$n_{peak}$ (Normalized) or $Q_{total}(nC)$:', style=style, layout=layout)
    
    # Witness beam
    z_witnessW = widgets.FloatText(value=180, description='witness z position (Normalized/$\mu m$):', style=style, layout=layout)

    sigma_r_witnessW = widgets.FloatText(value=2.5, description='$\sigma_r$ (Normalized/$\mu m$):', style=style, layout=layout)
    epsilon_n_witnessW = widgets.FloatText(value=3.15, description='$\epsilon_n$ (Normalized/$\mu m$):', style=style, layout=layout)
    sigma_z_witnessW = widgets.FloatText(value=5, description='$\sigma_z$ (Normalized/$\mu m$):', style=style, layout=layout)

    gammaE_witnessW = widgets.FloatText(value=20000, description='$\gamma$:', style=style, layout=layout)    
    energy_spread_witnessW = widgets.FloatText(value=0.0025, description='$\Delta \gamma / \gamma$:', style=style, layout=layout) 
    peak_density_witnessW = widgets.FloatText(value=0.5, description='$n_{peak}$ (Normalized) or $Q_{total}(nC)$:', style=style, layout=layout)

    
    interact_calc(makeInput,inputDeckTemplateName = inputDeckTemplateNameW,units = unitsW,
                  indx = indxW,indz=indzW,n0 = n0W,time = timeW,dt = dtW,nbeams = nbeamsW,
                  boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
                  z_driver = z_driverW,
                  sigma_x_driver = sigma_x_driverW,sigma_y_driver = sigma_y_driverW,sigma_z_driver = sigma_z_driverW,  
                  gammaE_driver = gammaE_driverW,
                  peak_density_driver = peak_density_driverW,
                  z_witness = z_witnessW,
                  sigma_r_witness = sigma_r_witnessW,epsilon_n_witness = epsilon_n_witnessW,sigma_z_witness = sigma_z_witnessW,  
                  gammaE_witness = gammaE_witnessW,energy_spread_witness = energy_spread_witnessW,
                  peak_density_witness = peak_density_witnessW
                 );
