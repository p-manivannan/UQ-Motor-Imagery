from models import EnsembleModel, MCDropConnectModel, MCDropoutModel
from models import DropConnectModel, DropoutModel
from models import DUQModel, FlipoutModel

def condense_string(text):
    return text.strip().lower().replace(' ', '_')

def get_class(method):
    method = condense_string(method)
    if method is None:
        raise ValueError ("method provided is None!")
    if 'dropout' in method:
        if 'mc' in method:
            return MCDropoutModel()
        else:
            return DropoutModel()
    elif 'connect' in method:
        if 'mc' in method:
            return MCDropConnectModel()
        else:
            return DropConnectModel()
    elif 'ensem' in method:
        return EnsembleModel()
    elif 'duq' in method:
        return DUQModel()
    elif 'flip' in method:
        return FlipoutModel()
    return None

'''
returns true if a model is stochastic
(MC methods and flipout)
'''
def isMethodStochastic(method):
    return True if 'mc' in method or 'flipout' in method else False

'''
Gets method to tune for based on method name.
Case 1: If given method is either: ensembles, MC-Dropout or Dropout,
dropout will be tuned/trained.
Case 2: If given method is either MC-DropConnect or DropConnect,
DropConnect will be tuned/trained
Case 3: Otherwise, the tuning based method is the same as the method
(DUQ, Flipout)
This is because the hyperparameters are the same for all methods in Case 1.
The same can be said of Case 2. So there is no needless tuning.
'''
def alias_method(method):
    method = condense_string(method)
    if method is None:
        raise ValueError ("method provided is None!")
    if 'dropout' in method or 'ensemble' in method:
        return 'dropout'
    elif 'connect' in method:
        return 'dropconnect'
    return method

def determine_hypermodel(method):
    Class = get_class(method)
    return Class.build