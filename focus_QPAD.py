# This notebook should be placed in the 'Fields' folder
# Users should run 'merge_modes.py' to calculate the total Er, Ephi, Br, Bphi from different modes
# Then this scripts calculate Fr = Er - Bphi, Ephi = Ephi + Br
# The user should set the angle (the first line of code) correctly

import numpy as np
import h5py
import os
import glob

angle = 0

def get_file_list():
    file_list = glob.glob('Er/Merged_angle_' + str(angle) + '/*.h5')
    file_list = [os.path.basename(file) for file in file_list]
    return file_list

file_list = get_file_list()
postfixs = [i.split('_')[1] for i in file_list]

# Extract the attributes of dataset and axes, then modify the attributes slightly
with h5py.File('Ephi/Merged_angle_' + str(angle) + '/ephi_00000000.h5','r') as h5file:
    # Extract the dataset from the file
    dset_name = list(h5file.keys())[1]
    h5dset = h5file[dset_name]

    # Modify dataset's attributes
    dset_attrs = {name: value for name, value in h5dset.attrs.items()}
    dset_attrs['LONG_NAME'] = np.array([b'F_r'], dtype='|S100')
    # Read axes and their attributes
    h5axis1 = h5file['AXIS']['AXIS1']
    h5axis2 = h5file['AXIS']['AXIS2']
    axis1_attrs = {name: value for name, value in h5axis1.attrs.items()}
    axis2_attrs = {name: value for name, value in h5axis2.attrs.items()}
    axis1 = np.array(h5axis1)
    axis2 = np.array(h5axis2)
    
# Calculate Fr = Er - Bphi
for postfix in postfixs:

    filename_Er = 'Er/Merged_angle_' + str(angle) + '/er_'+ postfix
    filename_Bphi = 'Bphi/Merged_angle_' + str(angle) + '/bphi_' + postfix
    with h5py.File(filename_Er,'r') as h5file:
        # Modify the file's attributes
        file_attrs = {name: value for name, value in h5file.attrs.items()}
        file_attrs['NAME'] = np.array([b'fr'], dtype='|S100')
        # read dataset name and attributes
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Er = np.array(h5dset)
    with h5py.File(filename_Bphi,'r') as h5file:
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Bphi = np.array(h5dset)

    Fr = Er - Bphi
    if not os.path.exists('Fr'):
        os.mkdir('Fr')
    if not os.path.exists('Fr/Merged_angle_'+ str(angle)):
        os.mkdir('Fr/Merged_angle_'+ str(angle))
    # write to the h5 files
    print('Writing file: Fr/Merged_angle_'+ str(angle)+'/fr_' + postfix + '...')
    with h5py.File('Fr/Merged_angle_'+ str(angle)+'/fr_' + postfix, 'w') as h5file:
        # write file attributes
        for key, val in file_attrs.items():
            h5file.attrs[key] = val
        # create new dataset and write the attributes
        h5dset = h5file.create_dataset('fr', data=Fr, dtype='f8')
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
dset_attrs['LONG_NAME'] = np.array([b'F_\\phi'], dtype='|S100')            

# Calculate Fphi = Ephi + Br
for postfix in postfixs:

    filename_Ephi = 'Ephi/Merged_angle_' + str(angle) + '/ephi_'+ postfix
    filename_Br = 'Br/Merged_angle_' + str(angle) + '/br_' + postfix
    with h5py.File(filename_Ephi,'r') as h5file:
        # Modify the file's attributes
        file_attrs = {name: value for name, value in h5file.attrs.items()}
        file_attrs['NAME'] = np.array([b'fphi'], dtype='|S100')
        # read dataset name and attributes
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Ephi = np.array(h5dset)
    with h5py.File(filename_Br,'r') as h5file:
        dset_name = list(h5file.keys())[1]
        h5dset = h5file[dset_name]
        Br = np.array(h5dset)

    Fphi = Ephi + Br
    if not os.path.exists('Fphi'):
        os.mkdir('Fphi')
    if not os.path.exists('Fphi/Merged_angle_'+ str(angle)):
        os.mkdir('Fphi/Merged_angle_'+ str(angle))
    # write to the h5 files
    print('Writing file: Fphi/Merged_angle_'+ str(angle)+'/fphi_' + postfix + '...')
    with h5py.File('Fphi/Merged_angle_'+ str(angle)+'/fphi_' + postfix, 'w') as h5file:
        # write file attributes
        for key, val in file_attrs.items():
            h5file.attrs[key] = val
        # create new dataset and write the attributes
        h5dset = h5file.create_dataset('fphi', data=Fphi, dtype='f8')
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
