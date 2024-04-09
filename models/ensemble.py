from .standardmodels import DropoutModel
from keras_uncertainty.models import DeepEnsembleClassifier
from keras.layers import Dropout

class EnsembleModel(DropoutModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4, num_estimators=10):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)


    def get_model(self):
        if self.hp is not None:
            return DeepEnsembleClassifier(model_fn=lambda: self.build(self.hp), num_estimators=self.num_estimators) 
        else:
            raise ValueError("hp is none in get_model()!")
        
    def build(self, hp):
        return DeepEnsembleClassifier(model_fn=lambda: super.build(hp), num_estimators=self.num_estimators)


