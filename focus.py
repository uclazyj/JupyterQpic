# This script is used to post-process QuickPIC's results: It calculates the focusing field.
# It should be placed in the 'Fields' folder
# Assuming we have Ex, Ey, Bx, By in the 'Fields' folder, this scripts calculate Fx = Ex - By, Fy = Ey + Bx
# To run the script, simply executes: python3 focus.py in the terminal. Then the corresponding folders Fx_slice0001 and Fy_slice0002 should be generated. 

import numpy as np
import h5py
import os
import glob

def get_file_list():
    file_list = glob.glob('Ex_slice0001/*.h5')
    file_list = [os.path.basename(file) for file in file_list]
    return file_list

file_list = get_file_list()
postfixs = [i.split('_')[1] for i in file_list]

# Extract the attributes of dataset and axes, then modify the attributes slightly
with h5py.File('Ex_slice0001/exslicexz_00000000.h5','r') as h5file:
    # Extract the dataset from the file
    dset_name = list(h5file.keys())[1]
    h5dset = h5file[dset_name]

    # Modify dataset's attributes
    dset_attrs = {name: value for name, value in h5dset.attrs.items()}
    dset_attrs['LONG_NAME'] = np.array([b'F_x'], dtype='|S100')
    # Read axes and their attributes
    h5axis1 = h5file['AXIS']['AXIS1']
    h5axis2 = h5file['AXIS']['AXIS2']
    axis1_attrs = {name: value for name, value in h5axis1.attrs.items()}
    axis2_attrs = {name: value for name, value in h5axis2.attrs.items()}
    axis1 = np.array(h5axis1)
    axis2 = np.array(h5axis2)
    
# Calculate Fx = Ex - By
for postfix in postfixs:

    filename_Ex = 'Ex_slice0001/exslicexz_' + postfix
    filename_By = 'By_slice0001/byslicexz_' + postfix
    with h5py.File(filename_Ex,'r') as h5file:
        # Modify the file's attributes
        file_attrs = {name: value for name, value in h5file.attrs.items()}
        file_attrs['NAME'] = np.array([b'fx'], dtype='|S100')
        # read dataset name and attributes
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Ex = np.array(h5dset)
    with h5py.File(filename_By,'r') as h5file:
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        By = np.array(h5dset)

    Fx = Ex - By
    if not os.path.exists('Fx_slice0001'):
        os.mkdir('Fx_slice0001')
    # write to the h5 files
    print('Writing file: Fx_slice0001/fxslicexz_' + postfix + '...')
    with h5py.File('Fx_slice0001/fxslicexz_' + postfix, 'w') as h5file:
        # write file attributes
        for key, val in file_attrs.items():
            h5file.attrs[key] = val
        # create new dataset and write the attributes
        h5dset = h5file.create_dataset('fx', data=Fx, dtype='f8')
        for key, val in dset_attrs.items():
            h5dset.attrs[key] = val

        # create group for axis and write the informations
        h5axis = h5file.create_group('AXIS')
        h5axis1 = h5axis.create_dataset('AXIS1', data=axis1)
        h5axis2 = h5axis.create_dataset('AXIS2', data=axis2)
        for key, val in axis1_attrs.items():
            h5axis1.attrs[key] = val
        for key, val in axis2_attrs.items():
            h5axis2.attrs[key] = val

# Change the attributes for Fphi
dset_attrs['LONG_NAME'] = np.array([b'F_y'], dtype='|S100')            

# Calculate Fy = Ey + Bx
for postfix in postfixs:

    filename_Ey = 'Ey_slice0002/eysliceyz_' + postfix
    filename_Bx = 'Bx_slice0002/bxsliceyz_' + postfix
    with h5py.File(filename_Ey,'r') as h5file:
        # Modify the file's attributes
        file_attrs = {name: value for name, value in h5file.attrs.items()}
        file_attrs['NAME'] = np.array([b'fy'], dtype='|S100')
        # read dataset name and attributes
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Ey = np.array(h5dset)
    with h5py.File(filename_Bx,'r') as h5file:
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Bx = np.array(h5dset)

    Fy = Ey + Bx
    if not os.path.exists('Fy_slice0002'):
        os.mkdir('Fy_slice0002')
    # write to the h5 files
    print('Writing file: Fy_slice0002/fysliceyz_' + postfix + '...')
    with h5py.File('Fy_slice0002/fysliceyz_' + postfix, 'w') as h5file:
        # write file attributes
        for key, val in file_attrs.items():
            h5file.attrs[key] = val
        # create new dataset and write the attributes
        h5dset = h5file.create_dataset('fy', data=Fy, dtype='f8')
        for key, val in dset_attrs.items():
            h5dset.attrs[key] = val

        # create group for axis and write the informations
        h5axis = h5file.create_group('AXIS')
        h5axis1 = h5axis.create_dataset('AXIS1', data=axis1)
        h5axis2 = h5axis.create_dataset('AXIS2', data=axis2)
        for key, val in axis1_attrs.items():
            h5axis1.attrs[key] = val
        for key, val in axis2_attrs.items():
            h5axis2.attrs[key] = val
