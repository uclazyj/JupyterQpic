import h5py
import numpy as np
import matplotlib.pyplot as plt

import json
from collections import OrderedDict
from ipywidgets import interact,FloatSlider
import ipywidgets as widgets
from helper import *

import os

# Take numerical differentiation for a 1D numpy array
def NDiff1D(x,y):
    if len(x) != len(y):
        print('The length of the input array are not the same!')
        return
    if len(x) < 2:
        print('The length of the input array is less than 2!')
        return
    N = len(x)
    dydx = np.zeros(N)
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1,N-1):
        dydx[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dydx

# Take numerical differentiation for a 2D numpy array
def NDiff(a,xLength,yLength,Ddir):
    nRows = a.shape[0]
    nCols = a.shape[1]
    dx = xLength / (nCols - 1)
    dy = yLength / (nRows - 1)
    b = a.copy()
    if(Ddir == 'row'):
        b[:,0] = (a[:,1] - a[:,0]) / dx
        b[:,-1] = (a[:,-1] - a[:,-2]) / dx
        b[:,1:-1] = (a[:,2:]-a[:,0:-2])/ (2*dx)
    elif(Ddir == 'column'):
        b[0,:] = (a[1,:] - a[0,:]) / dy
        b[-1,:] = (a[-1,:] - a[-2,:]) / dy
        b[1:-1,:] = (a[2:,:]-a[0:-2,:])/ (2*dy)
    return b



# Show_theory = 'focus'
# DiffDir = 'r' or 'xi'

def makeplot(fileNameList,scaleList = [1],LineoutDir = None,Show_theory = None,DiffDir = None,specify_title = ''):
    
    # This is the first filename
    filename = fileNameList[0]
    # Depending on what kind of data we are plotting, the best range of the colorbar and lineout is different
    
    if('Species' in filename):
        colorBarDefaultRange = [-5,0]
        colorBarTotalRange = [-10,0]
        lineoutAxisDefaultRange = [-5,0]
        lineoutAxisTotalRange = [-40,0]
    elif('Beam' in filename):
        colorBarDefaultRange = [-10,0]
        colorBarTotalRange = [-50,0]
        lineoutAxisDefaultRange = [-40,0]
        lineoutAxisTotalRange = [-100,0]
    elif('Fields' in filename):
        colorBarDefaultRange = [-1,1]
        colorBarTotalRange = [-2,2]
        lineoutAxisDefaultRange = [-2,2]
        lineoutAxisTotalRange = [-3,3]
    else:
        colorBarDefaultRange = [-1,1]
        colorBarTotalRange = [-2,2]
        lineoutAxisDefaultRange = [-2,2]
        lineoutAxisTotalRange = [-3,3]
  
    for i in range(len(fileNameList)):
        f=h5py.File(fileNameList[i],'r')
        k=list(f.keys()) # k = ['AXIS', 'charge_slice_xz']
        DATASET = f[k[1]]
        if(i == 0):
            data = np.array(DATASET) * scaleList[0]
        else:
            data += np.array(DATASET) * scaleList[i]

    AXIS = f[k[0]] # AXIS is a group, which contains two datasets: AXIS1 and AXIS2

    LONG_NAME = DATASET.attrs['LONG_NAME']
    UNITS = DATASET.attrs['UNITS']

    title = LONG_NAME[0].decode('UTF-8')
    unit = UNITS[0].decode('UTF-8')

    figure_title = title + ' [$' + unit + '$]' 
    if(specify_title != ''):
        figure_title = specify_title

    #### Read the axis labels and the corresponding units

    AXIS1 = AXIS['AXIS1']
    AXIS2 = AXIS['AXIS2']

    LONG_NAME1 = AXIS1.attrs['LONG_NAME']
    UNITS1 = AXIS1.attrs['UNITS']

    LONG_NAME2 = AXIS2.attrs['LONG_NAME']
    UNITS2 = AXIS2.attrs['UNITS']

    axisLabel1 = LONG_NAME1[0].decode('UTF-8')
    unit1 = UNITS1[0].decode('UTF-8')

    axisLabel2 = LONG_NAME2[0].decode('UTF-8')
    unit2 = UNITS2[0].decode('UTF-8')

    label_bottom = '$'+axisLabel2+'$' + '  $[' + unit2 + ']$' 
    label_left = '$'+axisLabel1+'$' + '  $[' + unit1 + ']$' 

    axis = list(AXIS) # axis = ['AXIS1', 'AXIS2']

    xRange=np.array(f['AXIS/AXIS1'])
    xiRange=np.array(f['AXIS/AXIS2'])
    
    xLengthTotal = xRange[1] - xRange[0]
    zLengthTotal = xiRange[1] - xiRange[0]
    
    xCellsTotal = data.shape[1]
    zCellsTotal = data.shape[0]
    
    x=np.linspace(xRange[0],xRange[1],xCellsTotal)
    xi=np.linspace(xiRange[0],xiRange[1],zCellsTotal) 
    
    # Determine the range of the lineout, depending on the direction of the lineout
    lineoutRange = [0,0]
    if(LineoutDir == 'transverse'):
        lineoutRange = xiRange
    elif(LineoutDir == 'longitudinal'):
        lineoutRange = xRange / 2

    xLengthTotal = xRange[1] - xRange[0]
    zLengthTotal = xiRange[1] - xiRange[0]

    xCellsPerUnitLength = xCellsTotal/xLengthTotal
    zCellsPerUnitLength = zCellsTotal/zLengthTotal
   
    ##### If we need to take a derivative

    if(DiffDir == 'xi'):
        data = NDiff(data,xLengthTotal,zLengthTotal,Ddir = 'column')
    elif(DiffDir == 'r'):
        data = NDiff(data,xLengthTotal,zLengthTotal,Ddir = 'row')

    #####
    
    dataT = data.transpose()
    colormap = 'viridis'
    
    def plot(colorBarRange,lineoutAxisRange,lineout_position):  

        fig, ax1 = plt.subplots(figsize=(8,5))
        # Zoom in / zoom out the plot
        
        # The beam propagates to the left
        ax1.axis([ xi.min(), xi.max(),x.min()/2, x.max()/2])
        # The beam propagates to the right
        # ax1.axis([ xi.max(), xi.min(),x.min()/2, x.max()/2])
        
        ###

        ax1.set_title(figure_title)

        cs1 = ax1.pcolormesh(xi,x,dataT,vmin=colorBarRange[0],vmax=colorBarRange[1],cmap=colormap)
       
        fig.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(label_bottom)

        #Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(label_left, color='k')
        ax1.tick_params('y', colors='k')
        
        if(LineoutDir == 'longitudinal'):
            ax2 = ax1.twinx()
            middle_index = int(dataT.shape[0]/2)
            lineout_index = int (middle_index + lineout_position * xCellsPerUnitLength)
            lineout = dataT[lineout_index,:]
            ax2.plot(xi, lineout, 'r')
            
            if(Show_theory == 'focus'):
                # plot the 1/2 slope line (theoretical focusing force)
                focusing_force_theory = -1/2 * lineout_position * np.ones(dataT.shape[1])
                ax2.plot(xi,focusing_force_theory, 'r--',label='F = -1/2 r')
                ax2.legend()
            
    
            ax2.set_ylim(lineoutAxisRange)
            ax1.plot(xi, lineout_position*np.ones(dataT.shape[1]), 'b--') # Add a dashed line at the lineout position
            ax2.tick_params('y', colors='r')
        elif(LineoutDir == 'transverse'):
            ax2 = ax1.twiny()
            lineout_index = int ((lineout_position - xiRange[0] ) * zCellsPerUnitLength) 
            # prevent the user from selecting the last lineout position
            # which causes the index to be equal to the length of the array
            lineout_index = min(lineout_index,dataT.shape[1]-1)
            lineout = dataT[:,lineout_index]
            ax2.plot(lineout, x, 'r')
            
            if(Show_theory == 'focus'):
                # plot the 1/2 slope line (theoretical focusing force)
                focusing_force_theory = -1/2 * x
                ax2.plot(focusing_force_theory, x, 'r--',label='F = -1/2 r') 
                ax2.legend()
            
            ax2.set_xlim(lineoutAxisRange)
            ax1.plot(lineout_position*np.ones(dataT.shape[0]),x, 'b--')
            ax2.tick_params('x', colors='r')
            
            
        if(Show_theory == 'bubble_boundary'):
            boundary = getBubbleBoundary(filename)
            ax1.plot(boundary[0],boundary[1],'k--',label='bubble boundary')
            
#         if(Show_theory == 'Ez'):
#             boundary = getBubbleBoundary('rundir/Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5')
#             xii = boundary[0]
#             rb = boundary[1]
            
#             Delta_l = 1.1
#             Delta_s = 0.0
#             alpha = Delta_l/rb + Delta_s
#             beta = ((1+alpha)**2 * np.log((1+alpha)**2))/((1+alpha)**2-1)-1
            
#             psi0 = rb**2/4 * (1+beta)
#             Ez = Nd(xi,psi0)
#             Ez = smooth(Ez)
#             ax2.plot(xii,Ez,'r--',label='theory')
#             ax2.legend()
      
        fig.tight_layout()
     
        return
   
    i1=interact(plot,
                colorBarRange = widgets.FloatRangeSlider(value=colorBarDefaultRange,min=colorBarTotalRange[0],max=colorBarTotalRange[1],step=0.1,description='Colorbar:',continuous_update=False),
                lineoutAxisRange = widgets.FloatRangeSlider(value=lineoutAxisDefaultRange,min=lineoutAxisTotalRange[0],max=lineoutAxisTotalRange[1],step=0.1,description='lineout_axis_range:',continuous_update=False),
                lineout_position = FloatSlider(min=lineoutRange[0],max=lineoutRange[1],step=0.05,description='lineout position:',continuous_update=False)
               )
    return

# direction is chosen from 'transverse' or 'longitudinal'
# If 'transverse' is chosen, then 0 <= index < 2 ** indz
# If 'longitudinal' is chosen, then 0 <= index < 2 ** indx
# The return of the function are two 1D arrays. The first one is the axis, the second one is the lineout value
def getLineout(filename,direction,index):
    if(direction != 'transverse' and direction != 'longitudinal'):
        print('Wrong lineout direction!')
        return
    f=h5py.File(filename,'r')
    k=list(f.keys()) # k = ['AXIS', 'charge_slice_xz']
    DATASET = f[k[1]]
    data = np.array(DATASET)
    
    xRange=np.array(f['AXIS/AXIS1'])
    xiRange=np.array(f['AXIS/AXIS2'])
    
    zCellsTotal, xCellsTotal = data.shape
    
    if(direction == 'transverse'):
        if((index < 0) or (index >= zCellsTotal)):
            print('Wrong index!')
            return
        else:
            x=np.linspace(xRange[0],xRange[1],xCellsTotal)
            return (x,data[index,:])
            
    elif(direction == 'longitudinal'):
        if((index < 0) or (index >= xCellsTotal)):
            print('Wrong index!')
            return
        else:
            xi=np.linspace(xiRange[0],xiRange[1],zCellsTotal)
            return (xi,data[:,index])


def Nd(x,y):
    if(len(x)!=len(y)):
        print('Length of x and y have to be the same!')
        return
    if(len(x) < 5):
        print('Length of x is too short. Meaningless for numerical derivative!')
        return
    dy = np.zeros(len(x))
    dy[0] = (y[1]-y[0])/(x[1]-x[0])
    dy[1] = (y[2]-y[0])/(x[2]-x[0])
    dy[-1]=(y[-1]-y[-2])/(x[-1]-x[-2])
    dy[-2] = (y[-1]-y[-3])/(x[-1]-x[-3])
    for i in range(2,len(x)-2):
        dy[i] = (-y[i+2] + 8 * y[i+1] - 8 * y[i-1] + y[i-2])/12/((x[i+2]-x[i-2])/4)
    return dy


def smooth(x):
    if(len(x) < 7):
        return x
    x_smooth = np.zeros(len(x))
    x_smooth[0] = x[0]
    x_smooth[1] = (x[0] + x[1] + x[2])/3.0
    x_smooth[2] = (x[0] + x[1] + x[2] + x[3] + x[4])/5.0
    x_smooth[-1] = x[-1]
    x_smooth[-2] = (x[-1] + x[-2] + x[-3])/3.0
    x_smooth[-3] = (x[-1] + x[-2] + x[-3] + x[-4] + x[-5])/5.0
    for i in range(3,len(x) - 3):
        x_smooth[i] = (x[i-3] + x[i-2] + x[i-1] + x[i] + x[i+1] + x[i+2] + x[i+3])/7.0
    return x_smooth



# This function returns [xi_rb;rb_smoothed]
def getBubbleBoundary(filename,ionBubbleThreshold = -8e-2):
    
    f=h5py.File(filename,'r')
    k=list(f.keys())
    DATASET = f[k[1]]
    data = np.array(DATASET)
    xaxis=f['/AXIS/AXIS1'][...]
    yaxis=f['/AXIS/AXIS2'][...]

    xiCells = data.shape[0]
    xCells = data.shape[1]
    xMidIndex = int(xCells/2)

    xi=np.linspace(yaxis[0],yaxis[1],xiCells) 
    x=np.linspace(xaxis[0],xaxis[1],xCells) 

    axis = data[:,xMidIndex]

    rb = np.array([])
    xi_rb = np.array([])

    for i in range(xiCells):
        if(axis[i] > ionBubbleThreshold): # The part of axis inside the ion bubble
            xi_rb = np.append(xi_rb,xi[i])
            j = xMidIndex
            while((data[i][j] > ionBubbleThreshold) and (j < xCells)):
                j = j + 1
            rb = np.append(rb,x[j])

    rb_smooth = smooth(rb)
    rb_smooth = smooth(rb_smooth)
    boundary = np.array([xi_rb, rb_smooth])
    return boundary

# boundary = getBubbleBoundary()
# ax1.plot(boundary[0],boundary[1],'k--',label='bubble boundary')

def remove_outliers(x,p,remove_rate):
    if len(x) != len(p):
        print('The length of x and p must equal!')
        return
    if remove_rate < 0 or remove_rate > 1:
        print('The remove rate must between 0 and 1!')
        return
    N = len(x)
    idx = list(range(N))
    nRemove = int(N * remove_rate)
    # Sort the idx according to the absolute value of x. Remove the first nRemove indexes.
    idx_descending_abs_x = sorted(idx,key = lambda i:-abs(x[i]))
    idx_remove_x = set(idx_descending_abs_x[:nRemove])
    idx_descending_abs_p = sorted(idx,key = lambda i:-abs(p[i]))
    idx_remove_p = set(idx_descending_abs_p[:nRemove])
    idx_remove = idx_remove_x | idx_remove_p
    return np.array([(i not in idx_remove) for i in idx])

def analyze_raw_beam_data(timeSteps,beam_number = 2, zVisualizeCenter = 0, half_thickness = 0.1,QPAD = True):
    
    with open('../qpinput.json') as finput:
        inputDeck = json.load(finput,object_pairs_hook=OrderedDict)
    
    nbeams = inputDeck['simulation']['nbeams']
    if(beam_number > nbeams or beam_number <= 0):
        print('Invalid beam number!')
        return
    idx = int(beam_number-1)
    dt = inputDeck['simulation']['dt']
    
    zVisualizeMax = zVisualizeCenter + half_thickness
    zVisualizeMin = zVisualizeCenter - half_thickness
    
    emitn_x_z = []
    emit_x_z = []
    emitn_y_z = []
    emit_y_z = []
    gammaE_z = []
    energySpread_z = []
    alpha_x_z = []
    alpha_y_z = []
    beta_x_z = []
    beta_y_z = []
    sigma_x_z = []
    sigma_y_z = []

    
    s = [i * dt for i in timeSteps]

    parameters = {}
    
#     for timeStep in timeSteps:
    for i in range(len(timeSteps)):
        
        timeStep = timeSteps[i]
        timeStep = str(timeStep).zfill(8)
        temp = "" if QPAD else "000"
        filename = '../Beam'+ temp + str(beam_number)+'/Raw/raw_' + timeStep + '.h5'
        f=h5py.File(filename,'r')
        
        dataset_x3 = f['/x3'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        z = dataset_x3[...] # type(data) outputs numpy.ndarray
        
        n_all_particles = len(z)
        inVisualizationRange = (z > zVisualizeMin) & (z < zVisualizeMax)
        z = z[inVisualizationRange]
        n_in_range_particles = len(z)
        
        if i % 10 == 0:
            print('In file '+ filename +', analyzing ' + \
                  str(round((n_in_range_particles / n_all_particles * 100),2)) + '% particles (' + \
                  str(n_all_particles) + ' particles in the whole beam)')
        
        dataset_q = f['/q']
        q = dataset_q[...]
        q = q[inVisualizationRange]
#         q = abs(q)
#         weights = q / np.sum(q)
        weights = abs(q)
        
        dataset_p3 = f['/p3'] 
        gammaE = dataset_p3[...] 
        gammaE = gammaE[inVisualizationRange]
        gammaE_bar, sigma_gammaE = get_mean_and_std(gammaE,weights)
        gammaE_z.append(gammaE_bar)
        energySpread_z.append(sigma_gammaE / gammaE_bar)
        
        
        
        dataset_p1 = f['/p1'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        px = dataset_p1[...] # type(data) outputs numpy.ndarray
        px = px[inVisualizationRange] # extract the part within the data visualization range
        
        xprime = px / gammaE
        E_xprime, sigma_xprime = get_mean_and_std(xprime,weights)
        xprime = xprime - E_xprime

        E_px, sigma_px = get_mean_and_std(px,weights)
        px = px - E_px
        
        
        
        dataset_p2 = f['/p2'] 
        py = dataset_p2[...]
        py = py[inVisualizationRange]
        
        yprime = py / gammaE
        E_yprime, sigma_yprime = get_mean_and_std(yprime,weights)
        yprime = yprime - E_yprime
        
        E_py, sigma_py = get_mean_and_std(py,weights)
        py = py - E_py
        
        
        
        dataset_x1 = f['/x1'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        x = dataset_x1[...] # type(data) outputs numpy.ndarray
        x = x[inVisualizationRange]
        E_x, sigma_x = get_mean_and_std(x,weights)
        x = x - E_x
        sigma_x_z.append(sigma_x)

        
        
        dataset_x2 = f['/x2'] 
        y = dataset_x2[...]
        y = y[inVisualizationRange]
        E_y, sigma_y = get_mean_and_std(y,weights)
        y = y - E_y
        sigma_y_z.append(sigma_y)
        
        
        
        E_xpx, _ = get_mean_and_std(x*px,weights)  
        E_ypy, _ = get_mean_and_std(y*py,weights)  
        E_xxprime, _ = get_mean_and_std(x*xprime,weights)  
        E_yyprime, _ = get_mean_and_std(y*yprime,weights) 

        
        emitn_x = np.sqrt(sigma_x ** 2 * sigma_px ** 2 - E_xpx ** 2)
        emitn_x_z.append(emitn_x)
        
        emit_x = np.sqrt(sigma_x ** 2 * sigma_xprime ** 2 - E_xxprime ** 2)
        emit_x_z.append(emit_x)
        
        emitn_y = np.sqrt(sigma_y ** 2 * sigma_py ** 2 - E_ypy ** 2)
        emitn_y_z.append(emitn_y)
        
        emit_y = np.sqrt(sigma_y ** 2 * sigma_yprime ** 2 - E_yyprime ** 2)
        emit_y_z.append(emit_y)
        
        alpha_x = -E_xxprime / emit_x
        alpha_x_z.append(alpha_x)
        
        alpha_y = -E_yyprime / emit_y
        alpha_y_z.append(alpha_y)
        
        beta_x = sigma_x ** 2 / emit_x
        beta_x_z.append(beta_x)
        
        beta_y = sigma_y ** 2 / emit_y
        beta_y_z.append(beta_y)
        
    parameters['epsilon_n_x'] = emitn_x_z
    parameters['epsilon_n_y'] = emitn_y_z
    parameters['epsilon_x'] = emit_x_z
    parameters['epsilon_y'] = emit_y_z
    parameters['alpha_x'] = alpha_x_z
    parameters['alpha_y'] = alpha_y_z
    parameters['beta_x'] = beta_x_z
    parameters['beta_y'] = beta_y_z
    parameters['sigma_x'] = sigma_x_z
    parameters['sigma_y'] = sigma_y_z
    parameters['energy'] = gammaE_z
    parameters['energy_spread'] = energySpread_z
    parameters['s'] = s

    return parameters

# remove_outliers(x,p,remove_rate):
def analyze_raw_beam_data_remove_outliers(timeSteps,beam_number = 2, zVisualizeCenter = 0, half_thickness = 0.1,QPAD = True, remove_rate = 0):
    
    with open('../qpinput.json') as finput:
        inputDeck = json.load(finput,object_pairs_hook=OrderedDict)
    
    nbeams = inputDeck['simulation']['nbeams']
    if(beam_number > nbeams or beam_number <= 0):
        print('Invalid beam number!')
        return
    idx = int(beam_number-1)
    dt = inputDeck['simulation']['dt']
    
    zVisualizeMax = zVisualizeCenter + half_thickness
    zVisualizeMin = zVisualizeCenter - half_thickness
    
    emitn_x_z = []
    emit_x_z = []
    gammaE_z = []
    energySpread_z = []
    alpha_x_z = []
    beta_x_z = []
    sigma_x_z = []

    
    s = [i * dt for i in timeSteps]

    parameters = {}
    
#     for timeStep in timeSteps:
    for i in range(len(timeSteps)):
        
        timeStep = timeSteps[i]
        timeStep = str(timeStep).zfill(8)
        temp = "" if QPAD else "000"
        filename = '../Beam'+ temp + str(beam_number)+'/Raw/raw_' + timeStep + '.h5'
        f=h5py.File(filename,'r')
        
        dataset_x1 = f['/x1'] 
        x = dataset_x1[...]
        
        dataset_p1 = f['/p1'] 
        px = dataset_p1[...] 
        
        good_idx = remove_outliers(x,px,remove_rate) # good_idx are indexes of the particles who are not outliers
        
        x = x[good_idx]
        px = px[good_idx]
        
        dataset_q = f['/q']
        q = dataset_q[...]
        q = q[good_idx]
        
    
        dataset_x3 = f['/x3'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        z = dataset_x3[...] # type(data) outputs numpy.ndarray
        z = z[good_idx]
        
        dataset_p3 = f['/p3'] 
        gammaE = dataset_p3[...] 
        gammaE = gammaE[good_idx]
        
        n_all_particles = len(z)
        inVisualizationRange = (z > zVisualizeMin) & (z < zVisualizeMax)
        z = z[inVisualizationRange]
        n_in_range_particles = len(z)
        
        if i % 10 == 0:
            print('In file '+ filename +', analyzing ' + \
                  str(round((n_in_range_particles / n_all_particles * 100),2)) + '% particles')
        
        
        q = q[inVisualizationRange]
        weights = abs(q)
        
        gammaE = gammaE[inVisualizationRange]
        gammaE_bar, sigma_gammaE = get_mean_and_std(gammaE,weights)
        gammaE_z.append(gammaE_bar)
        energySpread_z.append(sigma_gammaE / gammaE_bar)

        px = px[inVisualizationRange] # extract the part within the data visualization range
        
        xprime = px / gammaE
        E_xprime, sigma_xprime = get_mean_and_std(xprime,weights)
        xprime = xprime - E_xprime

        E_px, sigma_px = get_mean_and_std(px,weights)
        px = px - E_px
             
        x = x[inVisualizationRange]
        E_x, sigma_x = get_mean_and_std(x,weights)
        x = x - E_x
        sigma_x_z.append(sigma_x)
        
        E_xpx, _ = get_mean_and_std(x*px,weights)   
        E_xxprime, _ = get_mean_and_std(x*xprime,weights)  

        emitn_x = np.sqrt(sigma_x ** 2 * sigma_px ** 2 - E_xpx ** 2)
        emitn_x_z.append(emitn_x)
        
        emit_x = np.sqrt(sigma_x ** 2 * sigma_xprime ** 2 - E_xxprime ** 2)
        emit_x_z.append(emit_x)
        
        alpha_x = -E_xxprime / emit_x
        alpha_x_z.append(alpha_x)
        
        beta_x = sigma_x ** 2 / emit_x
        beta_x_z.append(beta_x)
        
    parameters['epsilon_n_x'] = emitn_x_z
    parameters['epsilon_x'] = emit_x_z
    parameters['alpha_x'] = alpha_x_z
    parameters['beta_x'] = beta_x_z
    parameters['sigma_x'] = sigma_x_z
    parameters['energy'] = gammaE_z
    parameters['energy_spread'] = energySpread_z
    parameters['s'] = s

    return parameters


def save_beam_analysis(beam_number,xi_s,parameters_xi_s,half_thickness):
    if len(xi_s) != len(parameters_xi_s):
        print('The length of the inputs do not match!')
        return
    xi_s = [round(i,1) for i in xi_s]
    half_thickness = round(half_thickness,1)
    dic = {}
    for i in range(len(xi_s)):
        dic[xi_s[i]] = parameters_xi_s[i]
    filename = 'beam' + str(beam_number) + '_' + str(xi_s).replace(" ","") + '_' + str(half_thickness)
    with open(filename,'w') as f:
        json.dump(dic,f,indent=4)

def get_mean_and_std(x,weights):
    weights = weights / np.sum(weights) # normalize the weights so they sum to 1
    
    if len(x) == 0:
        print('The input array is empty!')
        return
    if len(x) != len(weights):
        print('The length of the input array and the length of the weights do not match!')
        return
    E_X = np.dot(x,weights)
    x = x - E_X
    sigma = np.sqrt(np.dot(x**2, weights))
    return (E_X, sigma)

def plot_phase_space(beam_number,xi_s,half_thickness_slice,timeSteps,xlim = None, ylim = None,dir_save = 'Phase_space',remove_rate = 0):
    
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)
        
    dt = get_one_item(['simulation','dt'])
    
    for i in range(len(timeSteps)):

        timeStep = timeSteps[i]

        filename = '../Beam'+ str(beam_number)+'/Raw/raw_' + str(timeStep).zfill(8) + '.h5'

        with h5py.File(filename, "r") as f:
            
            dataset_x1 = f['/x1'] 
            x_all = dataset_x1[...] 
            
            dataset_p1 = f['/p1'] 
            px_all = dataset_p1[...]
            
            good_idx = remove_outliers(x_all,px_all,remove_rate) # good_idx are indexes of the particles who are not outliers
            x_all = x_all[good_idx]
            px_all = px_all[good_idx]
            
            dataset_x3 = f['/x3'] 
            z_all = dataset_x3[...]
            z_all = z_all[good_idx]
            
        plt.figure(i)

        for xi in xi_s:
            zVisualizeMax = xi + half_thickness_slice
            zVisualizeMin = xi - half_thickness_slice
            inVisualizationRange = (z_all > zVisualizeMin) & (z_all < zVisualizeMax)
            x = x_all[inVisualizationRange]
            px = px_all[inVisualizationRange]
            plt.scatter(x,px,s=1,label = '$\\xi = $' + str(xi))

        plt.xlabel('$x$')
        plt.ylabel('$p_x$')
        if xlim != None:
            plt.xlim(xlim[0],xlim[1])
        if ylim != None:
            plt.ylim(ylim[0],ylim[1])
        plt.title('z = ' + str(int(dt * timeStep)))
        plt.legend(loc=(1.04,0))
        plt.savefig(dir_save + '/phase_space_'+str(timeStep).zfill(8)+'.png',bbox_inches = 'tight')
        plt.close()
    

#     gamma = inputDeck['beam'][idx]['gamma']
    
#     profile = inputDeck['beam'][idx]['profile']
#     if(profile == 0 or profile == 1):
#         sigma_z = inputDeck['beam'][idx]['sigma'][2]
#         sigma_x, sigma_y = inputDeck['beam'][idx]['sigma'][0], inputDeck['beam'][idx]['sigma'][1]
#         sigma_px, sigma_py = inputDeck['beam'][idx]['sigma_v'][0], inputDeck['beam'][idx]['sigma_v'][1]
#         alpha_ix, alpha_iy = 0,0
#         beta_ix, beta_iy = sigma_x ** 2 / (sigma_x * sigma_px / gamma), sigma_y ** 2 / (sigma_y * sigma_py / gamma)
#         sigma_gamma = inputDeck['beam'][idx]['sigma_v'][2]
#         energySpread = sigma_gamma / gamma
    
    
#     phi_bar_acc =  np.sqrt(2) * parameters['s'] / (np.sqrt(parameters['energy']) + np.sqrt(gamma))

#     sigma_phi = phi_bar_acc / 2 * sigma_gamma / np.sqrt(parameters['energy']) / np.sqrt(gamma)
                                                                    
#     parameters['emitn_x_theory_acc'] = np.sqrt(A_x**2 - (A_x**2-1) * np.exp(- 4 * sigma_phi ** 2))
#     parameters['emitn_y_theory_acc'] = np.sqrt(A_y**2 - (A_y**2-1) * np.exp(- 4 * sigma_phi ** 2))