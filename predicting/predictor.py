import file_functions as ff
from keras_uncertainty.models import DeepEnsembleClassifier, StochasticClassifier
import numpy as np
'''
Predictor should be able to handle the number of iterations
varying for different methods:
- DUQ, Dropout, DropConnect and ensembles need only 1 iteration
- Flipout, MC-Dropout and MC-Dropconnect require num_iterations
  prediction sets to average the effects of their stochasticity.

'''
class Predictor:
    def __init__(self, method=None, hp=None, forward_passes=50, num_iterations=50):
        # MC methods and their standard counterparts share the same weights. They are only different at prediction time
        self.method = method
        self.hp = hp
        self.forward_passes = forward_passes
        self.subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.wts_directory = ff.get_weights_directory(method)
        self.pred_directory = ff.get_predictions_directory(method)
        self.N = num_iterations
        self.predictions = {self.method : 
                            {'test': {'preds': [], 'labels': []},
                             'lockbox': {'preds':[], 'labels': []}}}

    def save_predictions(self, filename):
        filename = f'{ff.get_predictions_directory(self.method) + "/" + filename + ".h5"}'
        ff.safe_open_w(filename, 'w')      # Create directory
        ff.save_dict_to_hdf5(dic=self.predictions, filename=filename)

    '''
    Predicts and saves
    '''
    def predict(self, dataset, lockbox):
        if ff.isMethodStochastic(self.method):
            self.predict_n_passes(dataset, lockbox)
        else:
            self.predict_one_pass(dataset, lockbox)
            self.save_predictions(f'prediction')

    def empty_predictions_dict(self):
        self.predictions.clear()
        self.predictions = {self.method : 
                            {'test': {'preds': [], 'labels': []},
                             'lockbox': {'preds':[], 'labels': []}}}

    '''
    Passes referred to do not refer to forward passes,
    but re-prediction sets
    '''
    def predict_n_passes(self, dataset, lockbox):
        for n in range(self.N):
            self.predict_one_pass(dataset, lockbox)
            self.save_predictions(f'prediction_{n}')
            self.empty_predictions_dict()

    def predict_one_pass(self, dataset, lockbox):
        loaded_inputs = dataset['inputs']
        loaded_targets = dataset['targets']
        for test_subject_id in self.subject_ids:
            train_subj_ids = [x for x in self.subject_ids if x != test_subject_id]
            X_test = loaded_inputs[test_subject_id]
            Y_true = loaded_targets[test_subject_id]
            X_lock, Y_lock = ff.get_lockbox_data(loaded_inputs[train_subj_ids], loaded_targets[train_subj_ids], lockbox[test_subject_id])
            model = ff.get_class(self.method).build(self.hp)
            model.load_weights(f'{self.wts_directory + f"/test_subject_{test_subject_id}"}').expect_partial()
            Y_preds = self.predict_samples(X_test, model)
            lockbox_Y_preds = self.predict_samples(X_lock, model)
            self.append_to_predictions_dict(Y_preds, Y_true, lockbox_Y_preds, Y_lock)

        self.convert_preds_dict_2_numpy()
        

    '''
    Given a test set and a model, predicts samples to be returned
    '''
    def predict_samples(self, X, model):
        if ff.isMethodStochastic(self.method):
            model = StochasticClassifier(model)
            return model.predict_samples(X, self.forward_passes)
        elif 'ensemble' in self.method:
            model = DeepEnsembleClassifier(model_fn=lambda: ff.determine_hypermodel(self.method)(self.hp), num_estimators=10)
            return model.predict(X)
        else:
            return model.predict(X)


    def append_to_predictions_dict(self, Y_preds, Y_true, lockbox_Y_preds, Y_lock):
        for items in self.predictions.values():
            items['test']['preds'].append(Y_preds)
            items['test']['labels'].append(Y_true)
            items['lockbox']['preds'].append(lockbox_Y_preds)
            items['lockbox']['labels'].append(Y_lock)
    
    def convert_preds_dict_2_numpy(self):
        for items in self.predictions.values():
            items['test']['preds'] = np.array(items['test']['preds'])
            items['test']['labels'] = np.array(items['test']['labels'])
            items['lockbox']['preds'] = np.array(items['lockbox']['preds'])
            items['lockbox']['labels'] = np.array(items['lockbox']['labels'])

