"""Module predictions"""
import math
import os

import pandas as pd

import config
import src.data.pipelines as pipelines


class Predictions:
    """Class Predictions"""

    def __init__(self):
        """
        This constructor initialises the project's global variables
        """

        # Pipeline
        self.pipeline = pipelines.Pipelines()

        # Global variables
        variables = config.Config().variables()
        self.batch_size = variables['modelling']['batch_size']

    def images(self, data_set: pd.DataFrame):
        """
        Initialises the images pipeline, which reads-in images from a local directory, and converts
        each image to a tensor
        :param data_set: 
        :return: 
        """

        # Data
        tensors_generator_input = data_set.copy()
        tensors_generator_input.reset_index(inplace=True, drop=True)
        return self.pipeline.generator_tensorflow(tensors_generator_input), tensors_generator_input

    def evaluate(self, model, data_set_name, data_set, labels, network_checkpoints_path):
        tensors_of_images, tensors_generator_input = self.images(data_set)

        # Steps [Re-design: cf. steps_per_epoch, validation_steps in estimating.Estimating().parameters]
        steps = math.ceil(data_set.shape[0] / self.batch_size)

        # Predictions
        plausibilities = model.predict(tensors_of_images, steps=steps)
        predictor_output = pd.DataFrame(plausibilities, columns=labels)

        # Save
        tensors_generator_input.join(predictor_output).to_csv(os.path.join(network_checkpoints_path,
                                                                           data_set_name + '_' + 'predictions.csv'))
        return plausibilities
