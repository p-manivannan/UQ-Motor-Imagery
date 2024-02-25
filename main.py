import tuning as tn
from file_functions import load_dict_from_hdf5, load_lockbox

# Load method names
methods = ['standard_dropconnect', 'duq']
# Load data
dataset = load_dict_from_hdf5('dataset')
lockbox = load_lockbox()
# Tune, train, test and predict
for method in methods:
    Class = tn.get_class(method)
    tuner = tn.Tuner(method=method, hypermodel=Class.build)
    tuner.search(dataset, lockbox)
