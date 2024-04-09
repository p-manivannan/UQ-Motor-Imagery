from file_functions import *
from models_bachelors import *
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
from .model_utils import isMethodStochastic
from .file_functions import fetch_method_names

def create_per_subj_dict():
  # Create per-subject dictionary
  methods = fetch_method_names()
  metrics = fetch_metric_names()
  keys = fetch_keys()

  per_subj_dict = {}

  for method in methods:
      per_subj_dict[method] = {}
      for metric in metrics:
          per_subj_dict[method][metric] = {}
          for key in keys:
              per_subj_dict[method][metric][key] = []

  return per_subj_dict

'''
Checks if an np array is a standard method solely based
on its length.

Returns: 0 for not Standard, 1 for Standard mehods and 2 for DUQ

This is important because standard methods have different 
y_pred sizes and therefore require special consideration.
'''
def checkIfStandard(samples):
  return 1 if len(samples.shape) == 3 else 0

'''
Checks if a method is standard or not based on its method
name. 

Returns: 0 for not Standard, 1 for Standard mehods and 2 for DUQ

This is important because standard methods have different 
y_pred sizes and therefore require special consideration.
'''
def checkIfStandard(method):
   if method in ['dropout', 'dropconnect']:
      return 1
   elif 'duq' in method:
      return 2
   return 0 

'''
Mean of entropies of forward passes
Input shape: (9, 50, 576, 4)
Output shape: (9, 576)
'''
def shannon_entropy(samples, isStandard=0):
  if isStandard == 1:
    entropies = np.apply_along_axis(func1d=lambda x: x*np.log2(x), axis=-1,  arr=samples)
  elif isStandard == 2:
     return samples
  else:
    entropies = np.apply_along_axis(func1d=lambda x: x*np.log2(x), axis=-3, arr=samples).mean(axis=-3)

  return entropies.sum(axis=-1) * -1

'''
Entropies of means of forward 
Input shape: (9, 50, 576, 4)
'''
def predictive_entropy(samples, isStandard=0):
  if isStandard == 1:  # If standard model with no forward passes, then input shape is (9, 576, 4)
    entropies = np.apply_along_axis(func1d=lambda x: x*np.log2(x), axis=-1,  arr=samples)
  elif isStandard == 2:
     return samples
  else:
    entropies = samples.mean(axis=-3)
    entropies = np.apply_along_axis(func1d=lambda x: x*np.log2(x), axis=-1, arr=entropies)

  return entropies.sum(axis=-1) * -1
  

def mutual_information(samples, isStandard):
  return predictive_entropy(samples, isStandard) - shannon_entropy(samples, isStandard)

def normalize_entropy(entropy, n_classes=4):
  return entropy / np.log2(n_classes)

def normalize_information(info):
  return info / np.max(info) if np.max(info) != 0 else info

def predictive_uncertainty(samples, key, isStandard=0):
  entropy = predictive_entropy(samples, isStandard) if key == 'predictive-entropy' else shannon_entropy(samples, isStandard)
  norm = normalize_entropy(entropy)
  return norm

def get_uncertainty(y_pred, unc_method, isStandard=0):
    if isStandard == 2:     # For DUQ
       return y_pred.max(axis=-1)
    if unc_method == 'predictive-entropy':
        return predictive_uncertainty(y_pred, 'predictive-entropy', isStandard)
    elif unc_method == 'mutual-information':
        return normalize_information(mutual_information(y_pred, isStandard))
    elif unc_method == 'shannon-entropy':
        return predictive_uncertainty(y_pred, 'shannon-entropy', isStandard)


def get_corrects(Y_true, Y_pred, axis):
    if not checkIfStandard(Y_pred):
        Y_pred = np.mean(Y_true, axis=-3)       # averages forward passes if not already averaged
    return np.argmax(Y_true, axis=axis) == np.argmax(Y_pred, axis=axis)

# WIP
def load_predictions(method):
    if 'ensemble' in method:      # currently only ensemble based on regular dropout
        return load_dict_from_hdf5(f'ensemble/predictions/prediction.h5')
    elif 'duq' in method:
        return load_dict_from_hdf5(f'duq/predictions/prediction.h5')                     # Only cases are MC-Dropout and MC-DropConnect
    
    elif 'mc' in method or 'flipout' in method:
      reg = re.compile(r"\d+(?=\.)")
      if 'dropconnect' in method:
         directory = f'mcdropconnect/predictions'
      elif 'dropout' in method:
        directory = f'mcdropout/predictions'
      elif 'flipout' in method:
        directory = f'flipout/predictions' 
      num = max([int(reg.search(x).group()) for x in os.listdir(directory) if reg.search(x) != None]) + 1
      ret = {method: {'test': {'preds':[], 'labels':[]}, 'lockbox': {'preds':[], 'labels':[]}}}
      for n in range(num):
          temp_holder = load_dict_from_hdf5(directory + f'/prediction_{n}.h5')
          ret[method]['test']['preds'].append(temp_holder[method]['test']['preds'])
          ret[method]['lockbox']['preds'].append(temp_holder[method]['lockbox']['preds'])
          if n == 0:
            ret[method]['test']['labels'] = temp_holder[method]['test']['labels']
            ret[method]['lockbox']['labels'] = temp_holder[method]['lockbox']['labels']

      ret[method]['test']['preds'] = np.array(ret[method]['test']['preds'])
      ret[method]['lockbox']['preds'] = np.array(ret[method]['lockbox']['preds'])
      return ret

    elif 'mc' not in method:
      if 'dropout' in method:
        return load_dict_from_hdf5(f'dropout/predictions/prediction.h5')
      elif 'dropconnect' in method:
        return load_dict_from_hdf5((f'dropconnect/predictions/prediction.h5'))

def avg_forward_passes(data):
    data["preds"] = data["preds"].mean(axis=-3)
    return data

'''
For this to work in the general case 
(N_SUBJS, N_PRED_SETS, N_FORWARD_PASSES, N_TRIALS, N_CLASSES)
where the items are in any order, maybe can store a file
with this data that is created during pre-processing.
Stores N_SUBJS, N_TRIALS, N_CLASSES initially, and 
with each step of the pipeline, new info is added.

This file is read in during function calls like these
'''
def get_accuracies_helper(data):
   preds = data['preds']
   if len(preds.shape) < 3:
      return None
   elif len(preds.shape) == 3:   # 9, 576, 4. 
      return data
   elif len(preds.shape) == 4:   # 9, 50, 576, 4
      return avg_forward_passes(data)
   elif len(preds.shape) == 5:   # 50, 9, 50, 576, 4
      data['preds'] = data['preds'].mean(axis=0)
      return avg_forward_passes(data)
      
'''
TO-DO: Need to refactor this to work for 
9, 576, 4 or 9, 50, 576, 4 or 50, 9, 50, 576, 4
'''
def get_accuracies(data):
    data = get_accuracies_helper(data)
    acc = []
    y_preds = data["preds"].argmax(axis=-1)
    y_trues = data["labels"].argmax(axis=-1)
    
    # Get accuracy of each subject
    for idx, subject in enumerate(y_trues):
        score = accuracy_score(y_pred=subject, y_true=y_preds[idx], normalize=True)
        acc.append(score)
    
    return acc   
