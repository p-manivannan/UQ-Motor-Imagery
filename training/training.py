from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
import datetime

from models import EnsembleModel, MCDropConnectModel, MCDropoutModel
from models import DropConnectModel, DropoutModel
from models import DUQModel, FlipoutModel


'''
For each method, load tuner for that method.
Then reload the best results. Then train
with the best hyperparams.
'''

methods = load_methods()
for method in methods:
    hp = Tuner.load_best_model(method)
    trainer = Trainer(method, hp)
    trainer.train()
    

