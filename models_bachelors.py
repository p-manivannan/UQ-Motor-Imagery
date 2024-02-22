import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow import keras
import keras_tuner as kt
import numpy as np
import math
import keras.backend as K
from keras.constraints import max_norm
from keras_uncertainty.layers import DropConnectDense, StochasticDropout, RBFClassifier, FlipoutDense, FlipoutConv2D
from keras_uncertainty.layers import duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras_uncertainty.models import DeepEnsembleClassifier
import dropconnect_tensorflow as dc


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


# x_train_shape param required for flipout
# Function to create model depending on method
def create_model(method):
    if method == 'flipout':
        return build_flipout_model(load_tuned_flipout())
    elif method == 'duq':
        return build_duq_model(load_tuned_duq())
    elif 'dropout' in method or 'dropconnect' in method:
        dropout_hp, dropconnect_hp = load_tuned_models()
        if method == 'standard_dropout':
            return build_standard_model(dropout_hp)
        elif method == 'standard_dropconnect':
            return build_standard_model_dropconnect(dropconnect_hp)
        elif method == 'mcdropout':
            return build_dropout_model(dropout_hp)
        elif method == 'mcdropconnect':
            return build_dropconnect_model(dropconnect_hp)
        elif method == 'ensemble_dropout':
            return build_ensemble_model()   


def build_dropout_model(hp):

  # KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
  # NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
      C = 22          # Number of electrodes
      T = 1125        # Time samples of network input

      f_1 = 40              # F is number of convolutional kernels.
      k_1 = (1, 25)         # K is kernel size 

      # One thats size of channels and another that's size of timestamps

      f_2 = 40
      k_2 = (C, 1)

      f_p = (1, 75)   # Fp is pooling size
      s_p = (1, 15)   # Sp is pool stride

      Nc = 4          # Number of classes
      drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      conv_drop = hp.Boolean('conv_drop')
      # Another dropout layer before FC layer
      fc_drop = hp.Boolean('fc_drop')
      # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
      # ADD THE AXIS PARAM FOR MAX NORM
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                  activation="linear",
                                  input_shape = (C, T, 1),
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                  activation="linear",
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      if conv_drop:
        model.add(StochasticDropout(drop_rates_1))
      model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
      model.add(keras.layers.Activation(square))
      model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
      model.add(keras.layers.Activation(log))
      if fc_drop:
        model.add(StochasticDropout(drop_rates_2))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

      optimizer = keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=["accuracy"])
      return model

def build_dropconnect_model(hp):
  # KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
  # NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
      C = 22          # Number of electrodes
      T = 1125        # Time samples of network input

      f_1 = 40        # K is number of convolutional kernels.
      k_1 = (1, 25)        # F is kernel size 

      # One thats size of channels and another that's size of timestamps

      f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
      k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

      f_p = (1, 75)   # Fp is pooling size
      s_p = (1, 15)   # Sp is pool stride

      Nc = 4          # Number of classes
      drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
      conv_drop = hp.Boolean('conv_drop')
      # Another dropout layer before FC layer
      fc_drop = hp.Boolean('fc_drop')
      # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
      # ADD THE AXIS PARAM FOR MAX NORM
      model = keras.models.Sequential()
      model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                  activation="linear",
                                  input_shape = (C, T, 1),
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                  activation="linear",
                                  kernel_constraint = max_norm(2, axis=(0, 1, 2))))
      if conv_drop:
        model.add(DropConnectDense(22, drop_rates_1))
      model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
      model.add(keras.layers.Activation(square))
      model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
      model.add(keras.layers.Activation(log))
      if fc_drop:
        model.add(DropConnectDense(22, drop_rates_2))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

      optimizer = keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(loss="categorical_crossentropy",
                    optimizer=optimizer, metrics=["accuracy"])
      return model

def tune_model(x_train, x_val, y_train, y_val, method, callbacks):
    methods = {'mcdropout': build_dropout_model, 'mcdropconnect': build_dropconnect_model}
    tuner = kt.GridSearch(hypermodel=methods[method],
                          objective='val_loss',
                          max_trials=100,
                          executions_per_trial=1,
                          overwrite=True,
                          directory=f'{method}/tuning',
                          project_name=f'{method}')
    tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
                 callbacks=callbacks)
    return tuner

def build_standard_model(hp):
# KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
# NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
    C = 22          # Number of electrodes
    T = 1125        # Time samples of network input

    f_1 = 40        # K is number of convolutional kernels.
    k_1 = (1, 25)        # F is kernel size 

    # One thats size of channels and another that's size of timestamps

    f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
    k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

    f_p = (1, 75)   # Fp is pooling size
    s_p = (1, 15)   # Sp is pool stride

    Nc = 4          # Number of classes
    drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    conv_drop = hp.Boolean('conv_drop')
    # Another dropout layer before FC layer
    fc_drop = hp.Boolean('fc_drop')
    # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
    # ADD THE AXIS PARAM FOR MAX NORM
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                activation="linear",
                                input_shape = (C, T, 1),
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                activation="linear",
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    if conv_drop:
      model.add(keras.layers.Dropout(drop_rates_1))
    model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
    # Square activation
    model.add(keras.layers.Activation(lambda x: K.square(x)))
    model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
    # Log activation
    model.add(keras.layers.Activation(lambda x: K.log(K.clip(x, min_value = 1e-7, max_value = 10000)))) # lambda functions solve an error when trying to load weights for ensemble model
    if fc_drop:
      model.add(keras.layers.Dropout(drop_rates_2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model

def build_standard_model_dropconnect(hp):
# KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
# NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
    C = 22          # Number of electrodes
    T = 1125        # Time samples of network input

    f_1 = 40        # K is number of convolutional kernels.
    k_1 = (1, 25)        # F is kernel size 

    # One thats size of channels and another that's size of timestamps

    f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
    k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

    f_p = (1, 75)   # Fp is pooling size
    s_p = (1, 15)   # Sp is pool stride

    Nc = 4          # Number of classes
    drop_rates_1 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    drop_rates_2 = hp.Choice('drop_rates', [0.1, 0.2, 0.3, 0.4, 0.5])
    conv_drop = hp.Boolean('conv_drop')
    # Another dropout layer before FC layer
    fc_drop = hp.Boolean('fc_drop')
    # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
    # ADD THE AXIS PARAM FOR MAX NORM
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                activation="linear",
                                input_shape = (C, T, 1),
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                activation="linear",
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    if conv_drop:
      model.add(dc.DropConnectDense(units=22, prob=drop_rates_1, activation="linear", use_bias=False))
    model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
    model.add(keras.layers.Activation(square))
    model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
    model.add(keras.layers.Activation(log))
    if fc_drop:
      model.add(dc.DropConnectDense(units=22, prob=drop_rates_2, activation="linear", use_bias=False))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(Nc, activation='softmax', kernel_constraint = max_norm(0.5)))

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model

def build_flipout_model(hp):
    # Num batches is fixed during training from x_train_shape
    x_train_shape_0 = 3736
    num_batches = x_train_shape_0 / 32
    kl_weight = 1.0 / num_batches
    prior_sigma_1 = hp.Choice('prior_sigma_1', [1.0, 2.5, 5.0])
    prior_sigma_2 = hp.Choice('prior_sigma_2', [1.0, 2.5, 5.0])
    prior_pi = hp.Choice('prior_pi', [0.1, 0.25, 0.5])
    # KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
    # NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
    C = 22          # Number of electrodes
    T = 1125        # Time samples of network input

    f_1 = 40        # K is number of convolutional kernels.
    k_1 = (1, 25)        # F is kernel size 

    # One thats size of channels and another that's size of timestamps

    f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
    k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

    f_p = (1, 75)   # Fp is pooling size
    s_p = (1, 15)   # Sp is pool stride

    Nc = 4          # Number of classes
    n_units_dense = hp.Choice('n_units_dense', [10, 25, 50])
    n_units_2 = hp.Choice('n_units_2', [10, 25, 50])

    # I APPARENTLY SWITCH AROUND THE ORDER OF F1 AND K1
    # ADD THE AXIS PARAM FOR MAX NORM
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                activation="linear",
                                input_shape = (C, T, 1),
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                activation="linear",
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
    # Square activation
    model.add(keras.layers.Activation(lambda x: K.square(x)))
    model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
    # Log activation
    model.add(keras.layers.Activation(lambda x: K.log(K.clip(x, min_value = 1e-7, max_value = 10000)))) # lambda functions solve an error when trying to load weights for ensemble model
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_units_dense, activation='relu', kernel_constraint = max_norm(0.5)))
    model.add(FlipoutDense(n_units_2, kl_weight, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, bias_distribution=False, activation='relu'))
    model.add(FlipoutDense(Nc, kl_weight, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, bias_distribution=False, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model

def build_duq_model(hp):
# KERNEL SIZE IS HOW LONG SEGMENT IS THAT YOU'RE LOOKING AT. 
# NUMBER OF FILTERS IS HOW MANY OF THESE SEGMENTS YOU'RE LEARNING IT IS ARBITRARY
    C = 22          # Number of electrodes
    T = 1125        # Time samples of network input

    f_1 = 40        # K is number of convolutional kernels.
    k_1 = (1, 25)        # F is kernel size 

    # One thats size of channels and another that's size of timestamps

    f_2 = 40    # ITS SUPPOSED TO BE OTHER WAY AROUND. FILTER SIZE IS 
    k_2 = (C, 1)     # KERNEL SIZE SHOULD ACTUALLY BE C 

    f_p = (1, 75)   # Fp is pooling size
    s_p = (1, 15)   # Sp is pool stride

    Nc = 4          # Number of classes
    dense_units = hp.Choice('dense_units', [100, 200])
    activ = hp.Choice('dense_activation', ['relu'])
    centr_dims = hp.Choice('centroid_dims', [2, 5, 25, 100])
    length_scale = hp.Choice('length_scale', [0.1, 0.2, 0.3, 0.4, 0.5])
    train_centroids = hp.Choice('train_centroids', [False, True])
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=f_1,  kernel_size=k_1, padding = 'SAME',
                                activation="linear",
                                input_shape = (C, T, 1),
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.Conv2D(filters=f_2,  kernel_size=k_2, padding = 'SAME',
                                activation="linear",
                                kernel_constraint = max_norm(2, axis=(0, 1, 2))))
    model.add(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
    # Square activation
    model.add(keras.layers.Activation(square))
    model.add(keras.layers.AveragePooling2D(pool_size= f_p, strides= s_p))
    # Log activation
    model.add(keras.layers.Activation(log)) # lambda functions solve an error when trying to load weights for ensemble model
    model.add(keras.layers.Flatten())
    # LENGTH SCALE (0.1) IN THIS CASE IMPORTANT TO TUNE.
    # https://arxiv.org/pdf/2003.02037.pdf
    # Did a grid search (0, 1] while keeping lambda 0 and pick value with highest val acc.
    model.add(keras.layers.Dense(dense_units, activation=activ, kernel_constraint = max_norm(0.5)))
    model.add(RBFClassifier(Nc, length_scale, centroid_dims=centr_dims, trainable_centroids=train_centroids))

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # Tuned with binary cross entropy loss. Changed it to this because of a repo i saw:
    # https://github.com/Skiracer33/duq/blob/main/duq.ipynb
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, metrics=["categorical_accuracy"])
    
    # DUQ requires use of these 
    add_l2_regularization(model)
    return model

def build_ensemble_model():
  dropout_best_hps, dropconnect_best_hps = load_tuned_models()
  return DeepEnsembleClassifier(lambda: build_standard_model(dropout_best_hps), num_estimators=10) 

'''
Returns the best tuned models. 
For MCDropout this is:
{'drop_rates': 0.2, 'conv_drop': False, 'fc_drop': True}
For MCDropConnect this is:
{'drop_rates': 0.1, 'conv_drop': True, 'fc_drop': False}
Top models for MCDropConnect all do not have UQ layers.
'''
def load_tuned_models():
  mcdropout_tuner = kt.GridSearch(build_dropout_model,
                      objective='val_loss',
                      max_trials=100,
                      directory=f'mcdropout/tuning',
                      project_name=f'mcdropout')

  mcdropconnect_tuner = kt.GridSearch(build_dropconnect_model,
                      objective='val_loss',
                      max_trials=100,
                      directory=f'mcdropconnect/tuning',
                      project_name=f'mcdropconnect')

  mcdropout_tuner.reload()
  mcdropconnect_tuner.reload()
  # Dropout best params were index 0: 0.2 with only fc_drop
  # Dropconnect best params were index 5: 0.1 with only conv_drop
  # mcdropconnect_tuner.results_summary()
  dropout_best_hps = mcdropout_tuner.get_best_hyperparameters(num_trials=10)[0] # Top trial with any UQ layer
  dropconnect_best_hps = mcdropconnect_tuner.get_best_hyperparameters(num_trials=10)[5] # Top trial with any UQ layer
  return dropout_best_hps, dropconnect_best_hps

def load_tuned_duq():
  tuner = kt.GridSearch(hypermodel=build_duq_model,
                      objective='val_loss',
                      max_trials=200,
                      directory=f'duq/tuning',
                      project_name=f'9_february')
  tuner.reload()
  return tuner.get_best_hyperparameters(num_trials=1)[0]

def load_tuned_flipout():
  tuner = kt.GridSearch(hypermodel=build_flipout_model,
                        objective='val_loss',
                        max_trials=200,
                        executions_per_trial=1,
                        directory=f'flipout/tuning',
                        project_name=f'flipout_2')
  tuner.reload()
  return tuner.get_best_hyperparameters(num_trials=1)[0]