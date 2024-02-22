from .basemodel import BaseConvModel
from tensorflow import keras
from keras_uncertainty.layers import RBFClassifier, FlipoutDense
from keras_uncertainty.layers import add_l2_regularization
from keras.constraints import max_norm


class DUQModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dense(self, model):
        model.add(keras.layers.Dense(self.hp.Choice('n_units_dense', [100, 200]), 
                                          activation='relu', kernel_constraint = max_norm(0.5)))
        return model
    
    def add_rbf_layer(self, model):
        centr_dims = self.hp.Choice('centroid_dims', [2, 5, 25, 100])
        length_scale = self.hp.Choice('length_scale', [0.1, 0.2, 0.3, 0.4, 0.5])
        train_centroids = self.hp.Choice('train_centroids', [False, True])
        model.add(RBFClassifier(self.Nc, length_scale, centroid_dims=centr_dims, trainable_centroids=train_centroids))
        return model

    def build(self, hp):
        self.hp = hp
        model = keras.models.Sequential()
        self.add_conv_filters(model)
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.flatten(model)
        self.add_dense(model)
        self.add_rbf_layer(model)
        self.compile_model(model, loss='binary_crossentropy', metrics=["categorical_accuracy"])
        add_l2_regularization(model)
        return model

class FlipoutModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        x_train_shape_0 = 3736                      # Set after inspecting training data
        num_batches = x_train_shape_0 / 32
        self.kl_weight = 1.0 / num_batches          # Param fixed during training
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dense(self, model):
        model.add(keras.layers.Dense(self.hp.Choice('n_units_dense', [10, 25, 50]), 
                                          activation='relu', kernel_constraint = max_norm(0.5)))
        return model

    def add_flipout(self, model):
        prior_sigma_1 = self.hp.Choice('prior_sigma_1', [1.0, 2.5, 5.0])
        prior_sigma_2 = self.hp.Choice('prior_sigma_2', [1.0, 2.5, 5.0])
        prior_pi = self.hp.Choice('prior_pi', [0.1, 0.25, 0.5])
        n_units_2 = self.hp.Choice('n_units_2', [10, 25, 50])
        model.add(FlipoutDense(n_units_2, self.kl_weight, prior_sigma_1=prior_sigma_1, 
                                    prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, 
                                    bias_distribution=False, activation='relu'))
        model.add(FlipoutDense(self.Nc, self.kl_weight, prior_sigma_1=prior_sigma_1, 
                                    prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, 
                                    bias_distribution=False, activation='softmax'))
        return model
    
    def build(self, hp):
        self.hp = hp
        model = keras.layers.Sequential()
        self.add_conv_filters(model)
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.flatten(model)
        self.add_dense(model)
        self.add_flipout(model)
        self.compile_model(model)
        return model

