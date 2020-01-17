import numpy as np


class Confusion:

    def __init__(self):

        self.name = 'Confusion'


    @staticmethod
    def constraints(threshold, plausibilities):
        plausibilities = np.where(plausibilities > threshold, plausibilities, 0)

        maximum_per_record = (plausibilities == plausibilities.max(axis=1, keepdims=True, initial=0))

        return (maximum_per_record & (plausibilities > 0)).astype(int)


    def true_positive(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((truth == prediction) & (truth == 1)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return n_per_class


    def true_negative(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((truth == prediction) & (truth == 0)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return n_per_class


    def false_positive(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((prediction == 1) & (truth == 0)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return n_per_class


    def false_negative(self, threshold, plausibilities, truth):
        prediction = self.constraints(threshold, plausibilities)
        instances = ((prediction == 0) & (truth == 1)).astype(int)

        n_per_class = instances.sum(axis=0, keepdims=True).squeeze(axis=0).tolist()

        return n_per_class


    def calculate(self, plausibilities, truth, labels, path):

        # Temporary.  Delete.
        thresholds = np.arange(0, 0.9, 0.05)

        # Example
        [self.false_negative(i, plausibilities, truth) for i in thresholds]

        # Write to CSV
        print(labels)
        print(path)
