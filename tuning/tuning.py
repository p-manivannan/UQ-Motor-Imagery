from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
# from models_bachelors import *
import file_functions as ff
import tensorflow as tf
import keras_tuner as kt

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
'''
Tuner is built to tune the following:
- Dropout
- DropConnect
- DUQ
- Flipout
MC Methods, standard methods and Ensembling use the same
tuners. Hypermodel selection is done accordingly
'''
class Tuner:
    def __init__(self, n_epochs=200, callbacks=None, method=None, overwrite=False, objective='val_loss', max_trials=200,
                executions_per_trial=1, tuner_type='GridSearch'):
        self.n_epochs = n_epochs
        self.callbacks = self.default_callbacks() if callbacks is None else callbacks
        self.method = self.tuning_based_method(method)
        self.directory = f'{self.method}/tuning'
        self.overwrite = overwrite
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.tuner_type = tuner_type
        self.hypermodel = self.determine_hypermodel(self.method)
        self.subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.tuner = self.tuner_init()

    '''
    Gets method to tune for based on method name.
    Case 1: If given method is either: ensembles, MC-Dropout or Dropout,
    dropout will be tuned.
    Case 2: If given method is either MC-DropConnect or DropConnect,
    DropConnect will be tuned.
    Case 3: Otherwise, the tuning based method is the same as the method
    (DUQ, Flipout)
    This is because the hyperparameters are the same for all methods in Case 1.
    The same can be said of Case 2. So there is no needless tuning.
    '''
    def tuning_based_method(self, method):
        method = condense_string(method)
        if method is None:
            raise ValueError ("method provided is None!")
        if 'dropout' in method or 'ensemble' in method:
            return 'dropout'
        elif 'connect' in method:
            return 'dropconnect'
        return method

    def determine_hypermodel(self, method):
        Class = get_class(method)
        return Class.build

    def default_callbacks(self, patience=10, monitor='val_loss'):
        early_stopping = EarlyStopping(monitor=monitor, patience=patience)
        callbacks = [early_stopping]
        return callbacks

    def load_best_hps(self, num_trials=3, trial_num=0):
        tuner = self.tuner_init()
        tuner.reload()
        return tuner.get_best_hyperparameters(num_trials=num_trials)[trial_num]

    '''
    Only supports GridSearch for now
    Changes:
    - MC methods, their standard counterparts and ensembles share the same tuner,
      and the same HPs by extension. Take this into account when creating a tuner.
    '''
    def tuner_init(self):
        if self.tuner_type == 'GridSearch':
            tuner = kt.GridSearch(hypermodel=self.hypermodel,
                                  objective=self.objective,
                                  max_trials=self.max_trials,
                                  overwrite=self.overwrite,
                                  directory=self.directory)
            return tuner

    def search(self, dataset, lockbox, single_subj=True, subj=0):
        if single_subj:
            self.search_single_subject(dataset, lockbox, subj)
        else:
            ''' This condition is untested. Proceed with caution '''
            for n in self.subject_ids:
                self.search_single_subject(dataset, lockbox, n)

    def search_single_subject(self, dataset, lockbox, test_subject_id):
        lockbox_idxs = lockbox[test_subject_id]  # Get lockbox indexes of train set for test subject 0
        train_ids = self.subject_ids[:]
        train_ids.remove(test_subject_id)        # Remove test subject id from train subject ids
        loaded_inputs = dataset['inputs']
        loaded_targets = dataset['targets']
        train_inputs = loaded_inputs[train_ids]     # Inputs for training set
        train_targets = loaded_targets[train_ids]   # Targets for training set
        train_inputs, train_targets = ff.remove_lockbox(train_inputs, train_targets, lockbox_idxs)      # Remove lockbox data from train set
        X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, train_targets, test_size=0.1)
        self.tuner.search(X_train, Y_train, epochs=self.n_epochs, validation_data=(X_val, Y_val),
                        callbacks=self.callbacks)






