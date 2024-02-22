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
from result_analysis_functions import *
import tensorflow as tf
import keras_tuner as kt
from keras_uncertainty.models import DeepEnsembleClassifier

'''
Load best hyperparams
'''
# dropout_best_hps, dropconnect_best_hps = load_tuned_models()

n_epochs= 200
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

'''
model = DeepEnsembleClassifier(lambda: build_standard_model(dropout_best_hps), num_estimators=10) for ensemble
'''

methods = ['duq', 'flipout']
subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
'''
Load data
'''
dataset = load('all_subjects_runs_no_bandpass')
lockbox = load('lockbox')['data']
loaded_inputs = dataset['inputs']
loaded_targets = dataset['targets']

def load_hp(method):
    if method == 'flipout':
        return load_tuned_flipout()
    else:
        return load_tuned_duq()


'''
Training Loop
'''
for method in methods:
    directory = f'{method}/weights'
    # This loop leaves one subject for testing (denoted by the number in the name of the weights file).
    # Then it combines all the subject trials such that shape is now (8 * 576, 22, 1125).
    # Then selects 10% of this as the validation set. Then it trains diff. model on each set of train subjects.
    for test_subject_id in subject_ids:
        train_ids = subject_ids[:]
        train_ids.remove(test_subject_id)       # Remove test subject id
        test_subj_lockbox = lockbox[test_subject_id]        # Get lockbox indexes (8, 57) for the test subject
        inputs = loaded_inputs[train_ids]           # Get train set inputs
        targets = loaded_targets[train_ids]         # Get train set targets
        inputs, targets = remove_lockbox(inputs, targets, test_subj_lockbox)    # Remove lockboxed set from train set
        X_train, X_val, Y_train, Y_val = train_test_split(inputs, targets,test_size=0.1)
        model = create_model(method)
        history = model.fit(X_train, Y_train, epochs=n_epochs, validation_data=[X_val, Y_val],
                        callbacks=[early_stopping])
        model.save_weights(f'{directory}/test_subj_{test_subject_id}')