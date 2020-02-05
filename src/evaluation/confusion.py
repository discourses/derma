"""Module confusion"""
import os

import numpy as np
import pandas as pd

import config


class Confusion:
    """Class Confusion"""

    def __init__(self):
        """
        This constructor initialises global variables
        """
        variables = config.Config().variables()

        # Plausibility Thresholds
        thr = variables['evaluating']['thresholds']
        self.thresholds = np.arange(start=thr['min'], stop=thr['max'], step=thr['step'])

    @staticmethod
    def constraints(threshold: int, plausibilities):
        """

        :param threshold:
        :param plausibilities:
        :return:
        """
        plausibilities = np.where(plausibilities > threshold, plausibilities, 0)

        maximum_per_record = (plausibilities == plausibilities.max(axis=1, keepdims=True, initial=0))

        return (maximum_per_record & (plausibilities > 0)).astype(int)

    def true_positive(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((truth == prediction) & (truth == 1)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return [threshold] + n_per_class

    def true_negative(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((truth == prediction) & (truth == 0)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return [threshold] + n_per_class

    def false_positive(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((prediction == 1) & (truth == 0)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return [threshold] + n_per_class

    def false_negative(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((prediction == 0) & (truth == 1)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return [threshold] + n_per_class

    def case(self, plausibilities, truth, confusion_matrix_variable):
        return {
            'tn': [self.true_negative(i, plausibilities, truth) for i in self.thresholds],
            'fn': [self.false_negative(i, plausibilities, truth) for i in self.thresholds],
            'tp': [self.true_positive(i, plausibilities, truth) for i in self.thresholds],
            'fp': [self.false_positive(i, plausibilities, truth) for i in self.thresholds]
        }.get(confusion_matrix_variable,
              LookupError('{} could not be mapped to a function'.format(confusion_matrix_variable)))

    def evaluate(self, plausibilities, truth, labels, data_set_name,
                 network_checkpoints_path, confusion_matrix_variable):
        series = self.case(plausibilities, truth, confusion_matrix_variable)

        pd.DataFrame(series, columns=['thresholds'] + labels) \
            .to_csv(os.path.join(network_checkpoints_path, data_set_name + '_' + confusion_matrix_variable + '.csv'))
