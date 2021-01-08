import h5py
import numpy as np
import matplotlib.pyplot as plt

import json
from collections import OrderedDict
from ipywidgets import interact,FloatSlider
import ipywidgets as widgets

import os

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
    
    xLengthTotal = xRange[1] - xRange[0]
    zLengthTotal = xiRange[1] - xiRange[0]
    
    xCellsTotal = data.shape[1]
    zCellsTotal = data.shape[0]
    
    x=np.linspace(xRange[0],xRange[1],xCellsTotal)
    xi=np.linspace(xiRange[0],xiRange[1],zCellsTotal)
    
    if(direction == 'transverse'):
        if((index < 0) or (index >= zCellsTotal)):
            print('Wrong index!')
            return
        else:
            lineout = data[index,:]
            lineout = np.stack((x,lineout),axis=0)
            
    if(direction == 'longitudinal'):
        if((index < 0) or (index >= xCellsTotal)):
            print('Wrong index!')
            return
        else:
            lineout = data[:,index]
            lineout = np.stack((xi,lineout),axis=0)
    return lineout


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



###################### plot emittance for witness beam
# Normally, number = 1: drive beam; number = 2: witness beam
def analyze_beam_bata(ndump, last_file_number,first_file_number = 0,beam_number = 2, zVisualizeCenter = 0, halfThickness = 5):
    
    with open('../qpinput.json') as finput:
        inputDeck = json.load(finput,object_pairs_hook=OrderedDict)
    
    nbeams = inputDeck['simulation']['nbeams']
    if(beam_number > nbeams):
        return
    idx = int(beam_number-1)
    dt = inputDeck['simulation']['dt']
    zWitnessCenter = inputDeck['beam'][idx]['center'][2]
#     gamma = inputDeck['beam'][1]['gamma']
    
    profile = inputDeck['beam'][idx]['profile']
    if(profile == 0):
        sigmaz= inputDeck['beam'][idx]['sigma'][2]
#         sigma = inputDeck['beam'][1]['sigma'][0]
#         sigma_p = inputDeck['beam'][1]['sigma_v'][0]
#         emitn_i = sigma * sigma_p
#         alpha_i = 0
#         beta_i = sigma ** 2 / (emitn_i / gamma)
#         energySpread = inputDeck['beam'][1]['sigma_v'][2] / gamma
    elif(profile == 2):
        sigmaz= inputDeck['beam'][idx]['sigmaz']  
#         emitn_i = inputDeck['beam'][1]['emittance'][0]
#         alpha_i = inputDeck['beam'][1]['alpha'][0]
#         beta_i = inputDeck['beam'][1]['beta'][0]
#         energySpread = inputDeck['beam'][1]['sigma_vz'] / gamma
    
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

    timeSteps = range(first_file_number,last_file_number,ndump)
    s = np.array([i * dt for i in timeSteps])
    
#     # Calculate the theoretical emittance growth
#     gamma_i = (1 + alpha_i ** 2) / beta_i
#     beta_m = np.sqrt(2 * gamma) # normalized unit
#     A = (gamma_i * beta_m + beta_i / beta_m )/2
    
#     z_plasma = np.array(timeSteps) * dt
#     phi =  z_plasma / beta_m
#     emitEvolUniformPlasmaTheory = emitn_i * np.sqrt(A**2 - (A**2-1) * np.exp(-(energySpread * phi)**2))
    
    for timeStep in timeSteps:

        timeStep = str(timeStep).zfill(8)
        filename = '../Beam0002/Raw/raw_' + timeStep + '.h5'
        f=h5py.File(filename,'r')
        
        dataset_x3 = f['/x3'] # type(dataset) outputs: h5py._hl.dataset.Dataset
        z = dataset_x3[...] # type(data) outputs numpy.ndarray
        inVisualizationRange = (z > zVisualizeMin) & (z < zVisualizeMax)
#         inVisualizationRange = (z > float('-inf')) & (z < float('inf'))
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
        
    parameters = {}
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
#     # Plot the beam quantities according to the user's need
#     if(en):
#         fig, ax = plt.subplots()
#         plt.plot(timeSteps, emitn_x_z,label='x')
#         plt.plot(timeSteps, emitn_y_z,label='y')
# #         plt.plot(timeSteps, emitEvolUniformPlasmaTheory,label='Theory')
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\epsilon_n \;(c/\omega_p)$')
#         plt.legend(loc='lower right')
#         plt.show()
#         fig.savefig('emittance.png')
#     if(e):
#         plt.plot(timeSteps, emit_x_z,label='x')
#         plt.plot(timeSteps, emit_y_z,label='y')
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\epsilon \;(c/\omega_p)$')
#         plt.legend(loc='lower right')
#         plt.show()
#     if(a):
#         plt.plot(timeSteps, alpha_x_z,label='x')
#         plt.plot(timeSteps, alpha_y_z,label='y')
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\\alpha$')
#         plt.legend(loc='lower right')
#         plt.show()
#     if(b):
#         plt.plot(timeSteps, beta_x_z,label='x')
#         plt.plot(timeSteps, beta_y_z,label='y')
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\\beta \;(c/\omega_p)$')
#         plt.legend(loc='lower right')
#         plt.show()
    
#     if(sig):
#         plt.plot(timeSteps, sigma_x_z,label='x')
#         plt.plot(timeSteps, sigma_y_z,label='y')
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\sigma \;(c/\omega_p)$')
#         plt.legend(loc='lower right')
#         plt.show()
    
#     if(energy):
#         plt.plot(timeSteps, gammaE_z)
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\gamma $')
#         plt.show()
#     if(energySpread):
#         plt.plot(timeSteps, energySpread_z)
#         plt.xlabel('$z\;(c/\omega_p)$')
#         plt.ylabel('$\Delta \gamma /\gamma$ (%)')
#         plt.show()



