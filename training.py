from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models_bachelors import *
from file_functions import *
import tensorflow as tf
import keras_tuner as kt

def call_training_file(dataset, lockbox, loaded_inputs, loaded_targets):
    '''
    Load best hyperparams
    '''
    dropout_best_hps, dropconnect_best_hps = load_tuned_models()

    n_epochs= 100
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    methods = ['mcdropconnect', 'mcdropout']
    subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    '''
    Training Loop
    '''
    # Training loop for MCDropout and MCDropconnect methods
    for method in methods:
        directory = f'{method}/weights'
        # This loop leaves one subject for testing (denoted by the number in the name of the weights file).
        # Then it combines all the subject trials such that shape is now (8 * 576, 22, 1125).
        # Then selects 10% of this as the validation set. Then it trains diff. model on each set of train subjects.
        for test_subject_id in subject_ids:
            checkpoint_path = f'{directory}/test_subject_{test_subject_id}.ckpt'
            saving_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
            train_ids = subject_ids[:]
            train_ids.remove(test_subject_id)       # Remove test subject id
            test_subj_lockbox = lockbox[test_subject_id]        # Get lockbox indexes (8, 57) for the test subject
            inputs = loaded_inputs[train_ids]           # Get train set inputs
            targets = loaded_targets[train_ids]         # Get train set targets
            inputs, targets = remove_lockbox(inputs, targets, test_subj_lockbox)    # Remove lockboxed set from train set
            X_train, X_val, Y_train, Y_val = train_test_split(inputs, targets,test_size=0.1)
            
            model = build_dropout_model(dropout_best_hps) if method == 'mcdropout' else build_dropconnect_model(dropconnect_best_hps)
            history = model.fit(X_train, Y_train, epochs=n_epochs, validation_data=[X_val, Y_val],
                            callbacks=[early_stopping, saving_callback])
                    