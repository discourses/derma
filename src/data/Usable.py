import logging

import src.data.Sources as Sources
import configurations.configurations as cfg


class Usable:
    def __init__(self):
        # Logging
        cfg.logs()
        self.logger = logging.getLogger('debug')
        self.logger.disabled = True
        self.logger.name = __name__

    def summary(self):
        # The metadata & ground truth (listing), the names of the label columns, and
        # the names of the metadata fields
        listing, labels, fields = Sources.Sources().summary()

        # The minimum number of instances, i.e., data points, required
        # per class.  Presently based on the minimum number of instances per class
        # required for the stratification of imbalanced data sets w.r.t. SciKit Learn, etc..
        min_instances_per_class = 2

        # In terms of the data in question, how many instances are there per class?
        instances_per_class = listing[labels].sum()
        self.logger.info(f"Instances per class:\n\n{instances_per_class}")

        # Hence, are there any outlying classes?
        outliers = instances_per_class[instances_per_class < min_instances_per_class]
        self.logger.info(f"Outliers:\n\n{outliers}")

        # The inadmissible records, i.e., rows.  Herein, we are
        # determining the rows that belong to outlying classes.
        indices_of_inadmissible_rows = listing[listing[outliers.index.values].any(axis=1)].index

        # Hence, admissible
        admissible = listing.drop(indices_of_inadmissible_rows, axis=0).drop(outliers.index.values, axis=1)

        # The latest set of labels
        [labels.remove(i) for i in outliers.index.values]

        # If yes, the associated rows & columns are dropped.
        return admissible, labels, fields
