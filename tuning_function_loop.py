from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt

'''
Old mistake: I was training on test set and removing lockbox from test set.
'''

def call_tuning_file(dataset, lockbox, loaded_inputs, loaded_targets, methods):
    n_epochs= 100
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

    # Training loop for MCDropout and MCDropconnect models
    for method in methods:
        best_model = tune_model(X_train, X_val, Y_train, Y_val, method, callbacks)




