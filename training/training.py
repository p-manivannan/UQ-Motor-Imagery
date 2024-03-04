from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tuning import get_class


class Trainer:
    def __init__(self, method=None, hp=None, callbacks=None):
        self.method = method
        self.hp = hp
        self.callbacks = self.default_callbacks() if callbacks is None else callbacks
        self.subject_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.directory = f'{method}/weights'

    def default_callbacks(self, patience=10, monitor='val_loss'):
        early_stopping = EarlyStopping(monitor=monitor, patience=patience)
        saving_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          verbose=1)
        callbacks = [early_stopping, saving_callback]
        return callbacks

    def train(self, dataset, lockbox):
        for test_subject_id in self.subject_ids:
            checkpoint_path = f'{self.directory}/test_subject_{test_subject_id}'
            train_ids = subject_ids[:]
            train_ids.remove(test_subject_id)       # Remove test subject id
            test_subj_lockbox = lockbox[test_subject_id]        # Get lockbox indexes (8, 57) for the test subject
            loaded_inputs = dataset['inputs']
            loaded_targets = dataset['targets']
            inputs = loaded_inputs[train_ids]           # Get train set inputs
            targets = loaded_targets[train_ids]         # Get train set targets
            inputs, targets = remove_lockbox(inputs, targets, test_subj_lockbox)    # Remove lockboxed set from train set
            X_train, X_val, Y_train, Y_val = train_test_split(inputs, targets,test_size=0.1)
            
            model = get_class(self.method).build(self.hp)
            history = model.fit(X_train, Y_train, epochs=n_epochs, validation_data=[X_val, Y_val],
                            callbacks=self.callbacks)


