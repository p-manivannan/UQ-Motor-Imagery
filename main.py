import sys, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

import tuning as tn
import training as trn
import file_functions as ff

# Read method names from config file
methods = ['dropconnect', 'dropout', 'mcdropconnect', 'mcdropout', 'ensembles', 'flipout', 'duq']
# Load data
dataset = ff.load_dict_from_hdf5('dataset')
lockbox = ff.load_lockbox()
# Tune
# for method in methods:
#     Class = tn.get_class(method)
#     # Info in config file should allow skipping of redundant methods.
#     # Like tuning only either mcdropout or dropout. Same for
#     # mcdropconnect or dropconnect, and not tuning ensembles.
#     # Info in config file should be passed to Tuner to allow
#     # it to reject certain methods for tuning.
#     tuner = tn.Tuner(method=method, hypermodel=Class.build)
#     tuner.search(dataset, lockbox)

# Train
for method in methods:
    Class = tn.get_class(method)
    # Get best hps from reloaded tuners
    tuner = tn.Tuner(method=method)
    hp = tuner.load_best_hps()
    trainer = trn.Trainer(method=method, hp=hp)
    trainer.train(dataset, lockbox)


