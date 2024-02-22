from keras.backend import log, square, clip
from keras.constraints import max_norm
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam


class BaseConvModel:
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        self.C = C                               # Numbear of electrodes
        self.T = T                               # Time samples of network input
        self.f = f                               # Number of convolutional kernels
        self.k1 = k1                             # Kernel size
        self.k2 = (self.C, 1)                    # Kernel size
        self.fp = fp                             # Pooling size
        self.sp = sp                             # Pool stride
        self.Nc = Nc                             # Number of classes
        self.input_shape = (self.C, self.T, 1)
        self.hp = hp

    def add_conv_filters(self, model):
        model.add(Conv2D(filters=self.f,  kernel_size=self.k1, 
                                           padding = 'SAME',
                                           activation="linear",
                                           input_shape = self.input_shape,
                                           kernel_constraint = max_norm(2, axis=(0, 1, 2))))
        model.add(Conv2D(filters=self.f,  kernel_size=self.k2, 
                                           padding = 'SAME',
                                           activation="linear",
                                           kernel_constraint = max_norm(2, axis=(0, 1, 2))))
        return model
    
    def add_batch_norm(self, model):
        model.add(BatchNormalization(momentum=0.9, epsilon=1e-05))
        model.add(Activation(lambda x: square(x)))
        return model

    def add_pooling(self, model):
        model.add(AveragePooling2D(pool_size= self.fp, 
                                                     strides= self.sp))
        model.add(Activation(lambda x: log(clip(x, min_value = 1e-7, max_value = 10000))))
        return model

    def flatten(self, model):
        model.add(Flatten())
        return model

    def add_dense(self, model):
        model.add(Dense(self.Nc, activation='softmax', 
                                          kernel_constraint = max_norm(0.5)))
        return model
        
    def compile_model(self, model, lr=1e-4, loss='categorical_crossentropy', metrics=['accuracy']):
        optimizer = Adam(learning_rate=lr)
        model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
        return model
    
    def init_model(self):
        return Sequential()

    def build(self, hp):
        self.hp = hp
        model = self.init_model()
        self.add_conv_filters(model)
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.flatten(model)
        self.add_dense(model)
        self.compile_model(model)
        return model
    
    def get_model(self):
        return self.build()
    
