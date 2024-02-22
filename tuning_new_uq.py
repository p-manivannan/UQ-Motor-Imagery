from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

tf.config.list_physical_devices()

'''
Load data
'''
dataset = load('all_subjects_runs_no_bandpass')
lockbox = load('lockbox')['data']
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']

'''
n_epochs=200 for DUQ and flipout while 100 for rest
'''

n_epochs= 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
callbacks = [early_stopping]

subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
test_subject_id = 0
lockbox_idxs = lockbox[test_subject_id]  # Get lockbox indexes of train set for test subject 0
train_ids = subject_ids[:]
train_ids.remove(test_subject_id)        # Remove test subject id from train subject ids
train_inputs = loaded_inputs[train_ids]     # Inputs for training set
train_targets = loaded_targets[train_ids]   # Targets for training set
train_inputs, train_targets = remove_lockbox(train_inputs, train_targets, lockbox_idxs)      # Remove lockbox data from train set
X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, train_targets, test_size=0.1)

methods = {'duq': build_duq_model}
# methods = {'flipout': lambda x: build_flipout_model(x, X_train.shape[0])}

for method in methods:
    tuner = kt.GridSearch(hypermodel=methods[method],
                          objective='val_loss',
                          max_trials=n_epochs,
                          executions_per_trial=1,
                          overwrite=True,
                          directory=f'{method}/tuning',
                          project_name=f'9_february')
    tuner.search(X_train, Y_train, epochs=n_epochs, validation_data=(X_val, Y_val),
                 callbacks=callbacks)




