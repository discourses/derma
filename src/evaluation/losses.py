import os

import numpy as np
import pandas as pd

import config


class Losses:

    def __init__(self):
        # Variables
        variables = config.Config().variables()
        self.model_checkpoints_path = variables['modelling']['model_checkpoints_path']


    @staticmethod
    def series(history, network_checkpoints_path):
        array_of_metrics = np.array([history.epoch, history.history['loss'], history.history['val_loss']]).T

        metrics = pd.DataFrame(array_of_metrics, columns=['epoch', 'training_loss', 'validation_loss'])
        metrics['epoch'] = metrics.epoch.astype(int)

        metrics.to_csv(os.path.join(network_checkpoints_path, 'history.csv'), header=True, index=False)
