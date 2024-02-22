import keras
from .basemodel import BaseConvModel
import dropconnect_tensorflow as dc
from keras.models import Sequential
from keras.layers import Dropout

class DropConnectModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropconnect(self, model):
        model.add(dc.DropConnectDense(units=22, prob=self.hp.Choice('drop_rates',
                                           [0.1, 0.2, 0.3, 0.4, 0.5]), 
                                           activation="linear", use_bias=False))
        return model
    
    def build(self, hp):
        self.hp = hp
        fc_drop = self.hp.Boolean('fc_drop')
        conv_drop = self.hp.Boolean('conv_drop')
        model = Sequential()
        self.add_conv_filters(model)
        self.add_dropconnect(model) if conv_drop else None
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.add_dropconnect(model) if fc_drop else None
        self.flatten(model)
        self.add_dense(model)
        self.compile_model(model)
        return model


class DropoutModel(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dropout(self, model):
        model.add(Dropout(self.hp.Choice('drop_rates', 
                                            [0.1, 0.2, 0.3, 0.4, 0.5])))
        return model
    
    def build(self, hp):
        self.hp = hp
        fc_drop = self.hp.Boolean('fc_drop')
        conv_drop = self.hp.Boolean('conv_drop')
        model = Sequential()
        self.add_conv_filters(model)
        self.add_dropout(model) if conv_drop else None
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.add_dropout(model) if fc_drop else None
        self.flatten(model)
        self.add_dense(model)
        self.compile_model(model)
        return model

