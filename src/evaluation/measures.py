import typing

import pandas as pd

import config
import src.data.pipelines as pipelines
import src.evaluation.confusion as confusion
import src.evaluation.losses as losses
import src.evaluation.predictions as predictions


class Measures:

    def __init__(self):
        # Pipeline
        self.pipeline = pipelines.Pipelines()

        # Global variables
        variables = config.Config().variables()
        self.batch_size = variables['modelling']['batch_size']

        # Labels for the confusion matrix variables
        self.confusion_matrix_variables = variables['modelling']['confusion_matrix_variables']

        # Delivering to S3
        self.s3_path = variables['modelling']['s3_path']

    def calculate(self, history, network_checkpoints_path,
                  training_, validating_, testing_, labels):
        # losses
        losses.Losses().evaluate(history=history, network_checkpoints_path=network_checkpoints_path)

        # Data groups
        data_sets: typing.Dict[str, pd.DataFrame] = {'training': training_, 'validating': validating_,
                                                     'testing': testing_}

        # Hence
        for data_set_name, data_set in data_sets.items():
            # Raw predictions
            plausibilities = predictions.Predictions().evaluate(history.model, data_set_name, data_set[['url']],
                                                                labels, network_checkpoints_path)

            # Ground truth
            truth = data_set[labels].values

            # Confusion
            [confusion.Confusion().evaluate(plausibilities, truth, labels, data_set_name, network_checkpoints_path,
                                            confusion_matrix_variable)
             for confusion_matrix_variable in self.confusion_matrix_variables]
