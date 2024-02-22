import h5py
import os, os.path
import numpy as np

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')
        
def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_save_dict_contents_to_group( h5file, path, dic):

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")        

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, float, np.float32,int)):
            #print( 'here' )
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):            
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group( h5file, path): 

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans            


# Returns weights directory for a method
def get_weights_directory(method):
    if method in ['standard', 'standard_dropout', 'mcdropout']:
        return f'mcdropout/weights'
    elif method in ['dropconnect', 'standard_dropconnect', 'mcdropconnect']:
        return f'mcdropconnect/weights'
    else:
        return f'{method}/weights'
    
# checkpoint path is sometimes an hdf5 file in the case of ensembles and a ckpt file for mcdropout and the like.
# this function corrects the weight path depending on the extension
# duq, ensembles and flipout don't have .ckpt but mcdropout and mcdropconnect do
def rectify_wts_path(method, wts_path):
    if method in ['standard', 'standard_dropout', 'standard_dropconnect', 'mcdropout', 'mcdropconnect']:
        return wts_path + '.ckpt'
    else:
        return wts_path

# This function is used to to reshape np array shaped like
# (num_subjects, num_trials, num_channels, num_timestamps)
# for inputs and
# (num_subjects, num_trials, num_classes) for targets.
# to (total_num_trials, num_channels, num_timestamps)
# for input and you same for targets: Total trials in 
# first dimension.
def get_x_y(inputs, targets):
    n_subjects = inputs.shape[0]
    n_runs = inputs.shape[1] * n_subjects
    channels = inputs.shape[2]
    timestamps = inputs.shape[3]
    n_classes = targets.shape[2]
    X = np.vstack(inputs).reshape(n_runs, channels, timestamps)
    Y = np.vstack(targets).reshape(n_runs, n_classes)
    return X, Y

'''
Gets trials from lockbox indices (8, 57)
'''
def get_lockbox_data(loaded_inputs, loaded_targets, lockbox):
    inputs = loaded_inputs.copy()       # Shape: (9, 576, 22, 1125)
    targets = loaded_targets.copy()     # Shape: (9, 576, 4)
    per_sbj_lockbox_inputs = []
    per_sbj_lockbox_targets = []
    # Iterate through train set subject ids
    for i in range(lockbox.shape[0]):
        subj_inputs = inputs[i, lockbox[i,:], :, :]   
        subj_targets = targets[i, lockbox[i,:], :]
        per_sbj_lockbox_inputs.append(subj_inputs)
        per_sbj_lockbox_targets.append(subj_targets)
    
    return np.vstack(per_sbj_lockbox_inputs), np.vstack(per_sbj_lockbox_targets)



def remove_lockbox(loaded_inputs, loaded_targets, lockbox):
    inputs = loaded_inputs.copy()
    targets = loaded_targets.copy()
    per_sbj_lockbox_inputs = []
    per_sbj_lockbox_targets = []
    for i in range(lockbox.shape[0]):   # Iterate through each test subject and get lockboxed trials
        subj_inputs = inputs[i, lockbox[i,:], :, :]
        subj_targets = targets[i, lockbox[i,:], :]
        per_sbj_lockbox_inputs.append(subj_inputs)
        per_sbj_lockbox_targets.append(subj_targets)

    per_subj_keep_inputs = []
    per_subj_keep_targets = []
    for i in range(lockbox.shape[0]):   # Iterate through each test subject and delete lockboxed trials
        subj_keep_inputs = np.delete(inputs[i, :, :, :], obj=lockbox[i, :], axis=0)
        subj_keep_targets = np.delete(targets[i, :, :], obj=lockbox[i, :], axis=0)
        per_subj_keep_inputs.append(subj_keep_inputs)
        per_subj_keep_targets.append(subj_keep_targets)

    return np.vstack(per_subj_keep_inputs), np.vstack(per_subj_keep_targets)
    