import os
from typing import Dict

import pandas as pd

import src.data.pipelines as pipelines
import src.evaluation.confusion as confusion
import src.evaluation.losses as losses


class Measures:

    def __init__(self):

        # Labels for the confusion matrix variables
        self.confusion_matrix_variables = ['tn', 'fn', 'tp', 'fp']

        # Pipeline
        self.pipeline = pipelines.Pipelines()


    @staticmethod
    def confusion_variable_series(plausibilities, truth, labels, data_set_name,
                                  network_checkpoints_path, confusion_matrix_variable):

        series = confusion.Confusion().calculate(plausibilities, truth, confusion_matrix_variable)

        pd.DataFrame(series, columns=['thresholds'] + labels) \
            .to_csv(os.path.join(network_checkpoints_path, data_set_name + '_' + confusion_matrix_variable + '.csv'))


    def predictions(self, model, data_set_name, data_set, labels, network_checkpoints_path):

        # Data
        generator_input = data_set.copy()
        generator_input.reset_index(inplace=True, drop=True)
        tensors_of_images = self.pipeline.generator_tensorflow(generator_input)

        # Predictions
        plausibilities = model.predict(tensors_of_images)
        predictor_output = pd.DataFrame(plausibilities, columns=labels)

        # Save
        generator_input.join(predictor_output).to_csv(os.path.join(network_checkpoints_path,
                                                                   data_set_name + '_' + 'predictions.csv'))
        return plausibilities


    def calculate(self, history, network_checkpoints_path,
                  training_, validating_, testing_, labels):
        # losses
        losses.Losses().series(history=history, network_checkpoints_path=network_checkpoints_path)

        # Data groups
        data_sets: Dict[str, pd.DataFrame] = {'training': training_, 'validating': validating_, 'testing': testing_}

        # Hence
        for data_set_name, data_set in data_sets:

            # Raw predictions
            plausibilities = self.predictions(history.model, data_set_name, data_set[['url']],
                                              labels, network_checkpoints_path)

            # Ground truth
            truth = data_set[labels].values

            # Confusion
            [self.confusion_variable_series(plausibilities,
                                            truth,
                                            labels,
                                            data_set_name,
                                            network_checkpoints_path,
                                            confusion_matrix_variable)
             for confusion_matrix_variable in self.confusion_matrix_variables]
