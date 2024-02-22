'''
Flipout, MCDropout and MCDropConnect are stochastic models and hence
require 50 sets of predictions to approximate true performance
'''

'''
Trains the new UQ methods: ensembles, DUQ and Flipout.
Currently implementing and training them one at a time.
'''

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt
from keras_uncertainty.models import DeepEnsembleClassifier, StochasticClassifier


n_epochs= 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

'''
model = DeepEnsembleClassifier(lambda: build_standard_model(dropout_best_hps), num_estimators=10) for ensemble
'''

methods = ['duq']
subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
'''
Load data
'''
dataset = load('all_subjects_runs_no_bandpass')
lockbox = load('lockbox')['data']
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']

hp = load_tuned_duq()
# hp = load_tuned_flipout()

NUM = 1
# For each iteration, store results dict into a 
for iteration in range(0, NUM):
    # For each method, get preds and labels for each test subject
    # and their corresponding lockbox set.
    # predictions = {'standard': 
    #             {'test': {'preds':[], 'labels':[]}, 
    #                 'lockbox':{'preds':[], 'labels':[]}},
    #                 'standard_dropconnect': 
    #             {'test': {'preds':[], 'labels':[]}, 
    #                 'lockbox':{'preds':[], 'labels':[]}}
    #             }
    predictions = {'duq': 
                {'test': {'preds':[], 'labels':[]}, 
                    'lockbox':{'preds':[], 'labels':[]}}
                }
    # predictions = {'flipout': 
    #             {'test': {'preds':[], 'labels':[]}, 
    #                 'lockbox':{'preds':[], 'labels':[]}}
    #             }

    for method, values in predictions.items():
        print(f'{method}')
        if method == 'standard':
            wts_directory = f'mcdropout/weights'
        elif method == 'standard_dropconnect':
            wts_directory = f'mcdropconnect/weights'
        else:
            wts_directory = f'{method}/weights'
        # Iterate through test subjects
        for test_subject_id in range(0, 9):
            print(f'test subject {test_subject_id}')
            train_subj_ids = [x for x in subject_ids if x != test_subject_id]
            X_test = loaded_inputs[test_subject_id]
            Y_true = loaded_targets[test_subject_id]
            # Train set is sent in because lockbox is returned from the train set not the whole dataset.
            # This is because lockbox shape: (9, 8, 57) and inputs shape: (9, 576, 22, 1125)
            # Axis 0 are test_subj_ids and axis 1 are the train_subject_ids.
            # The function assumes that shape[0] of lockbox[test_subj_id] and shape[0] of
            # inputs is the same: 8.
            X_lock, Y_lock = get_lockbox_data(loaded_inputs[train_subj_ids], loaded_targets[train_subj_ids], lockbox[test_subject_id])
            wts_path = checkpoint_path = f'{wts_directory}/test_subj_{test_subject_id}'
            if method == 'mcdropout':
                model = build_dropout_model(dropout_best_hps)
            elif method == 'mcdropconnect':
                model = build_dropconnect_model(dropconnect_best_hps)
            elif method == 'standard_dropconnect':
                model = build_standard_model_dropconnect(dropconnect_best_hps)
            elif method == 'duq':
                model = build_duq_model(hp)
            elif method == 'flipout':
                model = build_flipout_model(hp) 
            else:
                model = build_standard_model(dropout_best_hps)
            
            model.load_weights(wts_path).expect_partial()
            # Get Y_preds for test subject
            if method in ['mcdropconnect', 'mcdropout']:
                model = StochasticClassifier(model)
                Y_preds = model.predict_samples(X_test, num_samples=50)
                # Get lockboxed Y_preds for test subject
                lockbox_Y_preds = model.predict_samples(X_lock, num_samples=50)
            elif method == 'flipout':
                model = StochasticClassifier(model)
                Y_preds = model.predict_samples(X_test, num_samples=50)
                # Get lockboxed Y_preds for test subject
                lockbox_Y_preds = model.predict_samples(X_lock, num_samples=50)

            else:
                Y_preds = model.predict(X_test)
                # Get lockboxed Y_preds for test subject
                lockbox_Y_preds = model.predict(X_lock)

            lockbox_Y_true = Y_lock
            values['test']['preds'].append(Y_preds)
            values['test']['labels'].append(Y_true)
            values['lockbox']['preds'].append(lockbox_Y_preds)
            values['lockbox']['labels'].append(lockbox_Y_true)

    for method, values in predictions.items():
        values['test']['preds'] = np.array(values['test']['preds'])
        values['test']['labels'] = np.array(values['test']['labels'])
        values['lockbox']['preds'] = np.array(values['lockbox']['preds'])
        values['lockbox']['labels'] = np.array(values['lockbox']['labels'])

    dict2hdf5(f'predictions/predictions_duq_new.h5', predictions)