import os
import logging

import pandas as pd

import config
import src.modelling.extraction.estimating as estimating
import src.modelling.extraction.hyperparameters as hyperparameters

import src.modelling.extraction.architecture as arc


class Steps:

    def __init__(self):

        # Logging
        config.Config().logs()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('root')
        self.logger.name = __name__

        # Variables
        variables = config.Config().variables()
        self.model_checkpoints_path = variables['modelling']['model_checkpoints_path']

    def cleanup(self):
        """
        Ensures that the hosting directories are empty
        :return:
        """

        files = [os.remove(os.path.join(base, file))
                 for base, directories, files in os.walk(self.model_checkpoints_path)
                 for file in files]

        directories = [os.removedirs(os.path.join(base, directory))
                       for base, directories, files in os.walk(self.model_checkpoints_path, topdown=False)
                       for directory in directories
                       if os.path.exists(os.path.join(base, directory))]

        if any(files) | any(directories):
            raise Exception(
                "Unable to delete all the items of {}".format(self.model_checkpoints_path))

    @staticmethod
    def partitions(path: str):
        """
        Creates a directory partition
        :param path: The directory that would be created if it doesn't exist
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)

    def evaluate(self, labels: list, epochs: int, hyperparameters_dict: dict,
                 training_: pd.DataFrame, validating_: pd.DataFrame, testing_: pd.DataFrame):
        """
        Runs one or more deep neural network models w.r.t. a defined architecture and sets of hyperparameters
        :param labels:  The names of the label columns
        :param epochs: The number of epochs
        :param training_: The training data, which includes the 'url' column of local image file paths
        :param validating_: The validation data, which includes the 'url' column of local image file paths
        :param testing_: The testing data, which includes the 'url' column of local image file paths
        :return:
        """

        # Ensure that the checkpoints directory is empty.  And, just in case the
        # checkpoints directory is deleted, re-create it
        self.cleanup()
        self.partitions(self.model_checkpoints_path)

        # A hyperparameters instance for creating a set of hyperparameters combinations
        hyp = hyperparameters.Hyperparameters(hyperparameters_dict=hyperparameters_dict)

        # A model estimation instance
        est = estimating.Estimating()

        # Estimate a model per hyperparameter combination
        for i in range(len(hyp.values())):

            # Ensure that the model checkpoints for a combination of hyperparameters are saved
            # to a distinct directory that EXISTS
            hyperparameters_set: str = str(i).zfill(4)
            network_checkpoints_path: str = os.path.join(self.model_checkpoints_path, hyperparameters_set)
            self.partitions(network_checkpoints_path)

            hyperparameters_set_values: dict = hyp.values()[i]
            self.logger.info(" Hyperparameters Set {}: {}".format(hyperparameters_set, hyperparameters_set_values))

            # Architecture, tf.python.keras.engine.sequential.Sequential,
            # tf.keras.models.Sequential, tf.keras.preprocessing.image.ImageDataGenerator
            architecture = arc.Architecture()
            model = architecture.layers(hyperparameters=hyperparameters_set_values, labels=labels)

            # History of losses
            est.network(model=model, labels=labels, epochs=epochs,
                        training_=training_, validating_=validating_, testing_=testing_,
                        network_checkpoints_path=network_checkpoints_path)
