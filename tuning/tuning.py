from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt
import datetime

from models import EnsembleModel, MCDropConnectModel, MCDropoutModel
from models import DropConnectModel, DropoutModel
from models import DUQModel, FlipoutModel

def condense_string(text):
    return text.strip().lower().replace(' ', '_')

def get_class(method):
    method = condense_string(method)
    if method is None:
        raise ValueError ("method provided is None!")
    if 'dropout' in method:
        if 'mc' in method:
            return MCDropoutModel()
        else:
            return DropoutModel()
    elif 'connect' in method:
        if 'mc' in method:
            return MCDropConnectModel()
        else:
            return DropConnectModel()
    elif 'ensem' in method:
        return EnsembleModel()
    elif 'duq' in method:
        return DUQModel()
    elif 'flip' in method:
        return FlipoutModel()
    return None

class Tuner:
    def __init__(self, n_epochs=200, callbacks=None, method=None, overwrite=True, objective='val_loss', max_trials=200,
                executions_per_trial=1, tuner_type='GridSearch', hypermodel=None):
        self.n_epochs = n_epochs
        self.callbacks = default_callbacks() if callbacks is None else callbacks
        self.directory = f'{self.method}/tuning'
        self.method = method
        self.overwrite = overwrite
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.project_name = self.determine_project_name()
        self.tuner_type = tuner_type
        self.hypermodel = hypermodel
        self.subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def determine_project_name(self):
        self.project_name = datetime.date.today()

    def default_callbacks(self, patience=10, monitor='val_loss'):
        early_stopping = EarlyStopping(monitor=monitor, patience=patience)
        self.callbacks = [early_stopping]

    '''
    Loads data from hdf5 file. Expects filename at the end
    '''
    def load_data(self, filename):
        return load_dict_from_hdf5(filename)

    '''
    Only supports GridSearch for now
    '''
    def tuner_init(self):
        if self.tuner_type == 'GridSearch':
            tuner = kt.GridSearch(hypermodel=self.hypermodel,
                                  objective=self.objective,
                                  max_trials=self.max_trials,
                                  overwrite=self.overwrite,
                                  directory=directory)
            return tuner

    def search(self, dataset, lockbox, single_subj=True):
        if single_subj:
            self.search_single_subject(dataset, lockbox, 0)

    def search_single_subject(self, dataset, lockbox, test_subject_id=0):
        lockbox_idxs = lockbox[test_subject_id]  # Get lockbox indexes of train set for test subject 0
        train_ids = subject_ids[:]
        train_ids.remove(test_subject_id)        # Remove test subject id from train subject ids
        loaded_inputs = dataset['inputs']
        loaded_targets = dataset['targets']
        train_inputs = loaded_inputs[train_ids]     # Inputs for training set
        train_targets = loaded_targets[train_ids]   # Targets for training set
        train_inputs, train_targets = remove_lockbox(train_inputs, train_targets, lockbox_idxs)      # Remove lockbox data from train set
        X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, train_targets, test_size=0.1)
        tuner = self.tuner_init()
        tuner.search(X_train, Y_train, epochs=self.n_epochs, validation_data=(X_val, Y_val),
                        callbacks=self.callbacks)



        


# subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# test_subject_id = 0
# lockbox_idxs = lockbox[test_subject_id]  # Get lockbox indexes of train set for test subject 0
# train_ids = subject_ids[:]
# train_ids.remove(test_subject_id)        # Remove test subject id from train subject ids
# train_inputs = loaded_inputs[train_ids]     # Inputs for training set
# train_targets = loaded_targets[train_ids]   # Targets for training set
# train_inputs, train_targets = remove_lockbox(train_inputs, train_targets, lockbox_idxs)      # Remove lockbox data from train set
# X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, train_targets, test_size=0.1)

# methods = {'duq': build_duq_model}
# # methods = {'flipout': lambda x: build_flipout_model(x, X_train.shape[0])}

# for method in methods:
#     tuner = kt.GridSearch(hypermodel=methods[method],
#                           objective='val_loss',
#                           max_trials=n_epochs,
#                           executions_per_trial=1,
#                           overwrite=True,
#                           directory=f'{method}/tuning',
#                           project_name=f'9_february')
#     tuner.search(X_train, Y_train, epochs=n_epochs, validation_data=(X_val, Y_val),
#                  callbacks=callbacks)

    




