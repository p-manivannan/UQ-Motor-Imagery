from .standardmodels import DropoutModel
from keras_uncertainty.models import DeepEnsembleClassifier
from keras.layers import Dropout

class EnsembleModel(DropoutModel):
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    # WIP
    # Q: How do I return an ensemble model when there is no tuned dropout?
    # Ans: Maybe check if the tuned dropout exists. If not, tune for dropout and set
    #      hyperparams of ensemble to dropout
    #      That's up to the tuning file to decide. That logic doesn't belong here

    def get_model(self, num_estimators=10):
        if self.hp is not None:
            return DeepEnsembleClassifier(model_fn=lambda: self.build(self.hp), num_estimators=num_estimators) 
        else:
            raise ValueError("hp is none in get_model()!")


