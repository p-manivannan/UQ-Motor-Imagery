from file_functions import *
from models_bachelors import *
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re

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
   if 'standard' in method:
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

def load_predictions(method, num=None):
    if 'standard' in method:        # Like standard_dropout/standard/standard_dropconnect
        return load_dict_from_hdf5(f'predictions/predictions_standard.h5')
    elif 'ensemble' in method:      # currently only ensemble based on regular dropout
        return load_dict_from_hdf5(f'predictions/predictions_ensemble_dropout.h5')
    elif 'duq' in method:
        return load_dict_from_hdf5(f'predictions/predictions_duq_new.h5')
    elif num != None:                           # Only cases are MC-Dropout and MC-DropConnect
        if 'standard' in method:
           return load_dict_from_hdf5(f'predictions/predictions_{num}.h5')
        else:   # Only flipout satisfies this condition for now
           return load_dict_from_hdf5(f'predictions/flipout_new/predictions_flipout_{num}.h5')
    else:
      reg = re.compile(r"\d+(?=\.)")
      directory = f'predictions/predictions_' if 'flipout' not in method else f'predictions/flipout_new/predictions_'
      if 'flipout' in method:
        num = max([int(reg.search(x).group()) for x in os.listdir('predictions/flipout') if reg.search(x) != None]) + 1
      else:
        num = max([int(reg.search(x).group()) for x in os.listdir('predictions') if reg.search(x) != None]) + 1
      ret = {method: {'test': {'preds':[], 'labels':[]}, 'lockbox': {'preds':[], 'labels':[]}}}
      for n in range(num):
          temp_holder = load_dict_from_hdf5(directory + f'{n}.h5')
          ret[method]['test']['preds'].append(temp_holder[method]['test']['preds'])
          ret[method]['lockbox']['preds'].append(temp_holder[method]['lockbox']['preds'])
          if n == 0:
            ret[method]['test']['labels'] = temp_holder[method]['test']['labels']
            ret[method]['lockbox']['labels'] =temp_holder[method]['lockbox']['labels']

      ret[method]['test']['preds'] = np.array(ret[method]['test']['preds'])
      ret[method]['lockbox']['preds'] = np.array(ret[method]['lockbox']['preds'])
      return ret
             

          
          

       
           
    

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

###########################################################################

'''The following is deprecated'''

###########################################################################
'''
takes as input a dict key: one of 'test' or 'lockbox'
'''
def avg_pred_entropy_plots(dataset, method, unc_method):
    bin_size = 0.05
    entropy_correct = []
    entropy_wrong = []
    N = 50

    # Iterate over all prediction sets. Also get the dropconnect predictions.
    for n in range(N):
        methods = load_predictions(n)
        data = methods[method]
        data = avg_forward_passes(data)
        if unc_method == 'predictive-normalised-entropy':
            entropy = predictive_uncertainty(data[dataset]['preds'])    # shape: (9,576)
        elif unc_method == 'mutual-information':
            entropy = mutual_information(data[dataset]['preds'])
        elif unc_method == 'shannon-entropy':
            entropy = shannon_entropy(data[dataset]['preds'])
        elif unc_method == 'predictive-entropy':
            entropy = predictive_entropy(data[dataset]['preds'])

        Y_true = data[dataset]['labels']    # shape: (9,576,4)
        corrects = get_corrects(Y_true, data[dataset]['preds'], axis=-1) # Get corrects across ALL subjects
        # Append the nth prediction's uncertainty estimations
        entropy_correct.append(entropy[corrects])
        entropy_wrong.append(entropy[~corrects])
        # For distribution plots of predictive entropy

    '''
    Check for data mismatch: entropy_correct is probably a list of np arrays instead of 
    1 cohesive np array 
    '''
    entropy_correct = np.hstack(entropy_correct)
    entropy_wrong = np.hstack(entropy_wrong)
    r = 5
    unc_cor = np.mean(entropy_correct)
    unc_cor_std = np.std(entropy_correct)
    unc_in = np.mean(entropy_wrong)
    unc_in_std = np.std(entropy_wrong)
    print(f"{dataset} avg. {unc_method} correct: {unc_cor:.5f} +/ {unc_cor_std:.5f}")
    print(f"{dataset} avg. {unc_method} wrong: {unc_in:.5f} +/ {unc_in_std:.5f}")

    # hist_data = [entropy_correct, entropy_wrong]    
    # group_labels = ['Correct', 'Incorrect']

    # # Normalizes AREA UNDER CURVE to sum up to 1. y-axis values are meaningless.
    # hist_correct, bins_correct, _ = plt.hist(entropy_correct, bins=20, density=True, alpha=0.5, label='Correct')
    # hist_wrong, bins_wrong, _ = plt.hist(entropy_wrong, bins=20, density=True, alpha=0.5, label='Wrong')
    # plt.legend()
    # # plt.show()

    # # Calculate overlap using histogram intersection
    # overlap = np.sum(np.minimum(hist_correct, hist_wrong))

    # # Normalize overlap between 0 and 1
    # normalized_overlap = overlap / np.sum(hist_correct)

    # print("Overlap:", normalized_overlap)
