from sklearn.model_selection import train_test_split
import file_functions as ff

'''
Predictor should be able to handle the number of iterations
varying for different methods:
- DUQ, Dropout, DropConnect and ensembles need only 1 iteration
- Flipout, MC-Dropout and MC-Dropconnect require num_iterations
  prediction sets to average the effects of their stochasticity.

'''
class Predictor:
    def __init__(self, method=None, hp=None, num_iterations=50):
        # MC methods and their standard counterparts share the same weights. They are only different at prediction time
        self.method = method
        self.hp = hp
        self.subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.directory = ff.get_weights_directory(method)
        self.N = num_iterations
        self.predictions = {self.method : 
                            {'test': {'preds': [], 'labels': []},
                             'lockbox': {'preds':[], 'labels': []}}}

    def save_predictions(self, filename):
        ff.save_dict_to_hdf5(f'{ff.get_predictions_directory(self.method) + filename}')

    def predict(self, dataset, lockbox):
        if ff.isMethodStochastic(self.method):
            self.predict_n_passes(dataset, lockbox)
        else:
            self.predict_one_pass(dataset, lockbox)
            self.save_predictions(f'prediction_{self.method}')


    def predict_n_passes(self, dataset, lockbox):
        pass

    def predict_one_pass(self, dataset, lockbox):
        pass

