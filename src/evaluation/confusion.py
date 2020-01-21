import numpy as np

import src.config as config


class Confusion:

    def __init__(self):

        variables = config.Config().variables()

        # Thresholds
        thr = variables['evaluating']['thresholds']
        self.thresholds = np.arange(start=thr['min'], stop=thr['max'], step=thr['step'])


    @staticmethod
    def constraints(threshold, plausibilities):
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


    def calculate(self, plausibilities, truth, variate):

        return {
            'tn': [self.true_negative(i, plausibilities, truth) for i in self.thresholds],
            'fn': [self.false_negative(i, plausibilities, truth) for i in self.thresholds],
            'tp': [self.true_positive(i, plausibilities, truth) for i in self.thresholds],
            'fp': [self.false_positive(i, plausibilities, truth) for i in self.thresholds]
        }.get(variate, LookupError('{} could not be mapped to a function'.format(variate)))
