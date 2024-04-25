import sys, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import pprint as pp

import tuning as tn
import training as trn
import predicting as prd
import file_functions as ff

# Read method names from config file
methods = ff.fetch_method_names()
# Load data
dataset = ff.load_dict_from_hdf5('dataset')
lockbox = ff.load_lockbox()
# Tune
# for method in methods:
#     Class = ff.get_class(method)
#     tuner = tn.Tuner(method=method, hypermodel=Class.build)
#     tuner.search(dataset, lockbox)

# # # Train
# for method in methods:
#     Class = ff.get_class(method)
#     # Get best hps from reloaded tuners
#     tuner = tn.Tuner(method=method)
#     hp = tuner.load_best_hps()
#     trainer = trn.Trainer(method=method, hp=hp)
#     trainer.train(dataset, lockbox)

# Predict
for method in methods:
    Class = ff.get_class(method)
    hp = tn.Tuner(method=method).load_best_hps()        # Check if best hps are being reloaded for DUQ
    predictor = prd.Predictor(method=method, hp=hp, forward_passes=2, num_iterations=2)
    predictor.predict(dataset, lockbox)



