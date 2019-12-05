import logging
import pandas as pd

import src.data.sources as Sources
import configurations.configurations as cfg


class Usable:
    """
    An extension of class Sources. It eliminates a class/label, and its records, if the class' sample size
    is < minimum_instances_per_class
    """

    # Preliminaries
    def __init__(self):
        """
        Only for logging states
        """

        # Logging
        cfg.logs()
        self.logger = logging.getLogger('debug')
        self.logger.disabled = True
        self.logger.name = __name__

    # The number of instances per class
    def instances_per_class(self, dataset: pd.DataFrame, labels: list):
        """
        This method calculates the number of instances/records per class w.r.t. a data set
        whose labels have been one-hot-coded.
        :param dataset: A data frame of data
        :param labels: The list of one-hot-coded label columns in dataset
        :return:
            summary: A pandas series that records the number of instances per class; it has a class names index.
        """

        # Calculate the number of instances/records per class
        summary: pd.Series = dataset[labels].sum()

        # Log the details
        self.logger.info("Instances per class:\n\n %s", summary)

        # Return
        return summary

    # Outlying classes
    def outlying_classes(self, instances_per_class: pd.Series):
        """
        A method that ...
        :param instances_per_class: A pandas series of
        :return:
            summary: outlying classes
        """

        # The minimum number of instances, i.e., data points, required
        # per class.  Presently based on the minimum number of instances per class
        # required for the stratification of imbalanced data sets w.r.t. SciKit Learn, etc.
        minimum_instances_per_class = cfg.variables()['modelling']['parameters']['minimum_instances_per_class']

        # Hence, are there any outlying classes?
        summary: pd.Series = instances_per_class[instances_per_class < minimum_instances_per_class]

        # Log the details
        self.logger.info("Outliers:\n\n %s", summary)

        return summary

    # The eliminator
    @staticmethod
    def summary():
        """
        Reads Sources.Sources().summary()
        :return:
            listing: Excludes classes that have a sample size < minimum_instances_per_class; all records associated
            with the excluded classes are deleted.
            labels: The classes remaining if any had to be excluded.
            fields: The unchanged metadata fields.
        """

        # The metadata & ground truth (listing), the names of the label columns, and
        # the names of the metadata fields
        listing, labels, fields = Sources.Sources().summary()

        # In terms of the data in question, how many instances are there per class?
        instances_per_class: pd.Series = Usable().instances_per_class(dataset=listing, labels=labels)

        # Hence, outlying classes
        outlying_classes: pd.Series = Usable().outlying_classes(instances_per_class)

        # The inadmissible records, i.e., rows.  Herein, we are
        # determining the rows that belong to outlying classes.
        i_inadmissible_records: pd.Int64Index = listing[listing[outlying_classes.index.values].any(axis=1)].index

        # Hence, admissible
        admissible = listing.drop(i_inadmissible_records, axis=0).drop(outlying_classes.index.values, axis=1)

        # The latest set of labels
        [labels.remove(i) for i in outlying_classes.index.values]

        # If yes, the associated rows & columns are dropped.
        return admissible, labels, fields
