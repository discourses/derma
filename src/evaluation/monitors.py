import tensorflow as tf

import os

import config


class Monitors:

    def __init__(self):

        # Variables
        variables = config.Config().variables()
        self.patience = variables['modelling']['early_stopping_patience']

    def early_stopping(self):

        # Early stopping
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', verbose=1, patience=self.patience, mode='min', restore_best_weights=True
        )

    @staticmethod
    def model_checkpoints(network_checkpoints_path: str):

        # Checkpoints
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(network_checkpoints_path, 'model_{epoch}.h5'),
            monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
            mode='auto', save_freq='epoch')
