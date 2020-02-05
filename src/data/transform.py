import logging
import os

import pandas as pd
import sklearn.model_selection as model_selection

import config


class Transform:

    def __init__(self):
        """
        Herein the constructor initialises global variables
        """

        # Root directory
        self.root = os.path.split(os.getcwd())[0]

        # Logging
        config.Config().logs()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('root')
        self.logger.name = __name__

        # Variables
        variables = config.Config().variables()
        self.random_state: int = variables['modelling']['random_state']
        self.train_size_initial: float = variables['modelling']['train_size_initial']
        self.train_size_evaluation: float = variables['modelling']['train_size_evaluation']

    @staticmethod
    def stratification(x: pd.DataFrame, y: pd.DataFrame,
                       train_size: float, random_state: int, stratify) -> (pd.DataFrame, pd.DataFrame,
                                                                           pd.DataFrame, pd.DataFrame):
        """
        Stratified splitting of a dataset

        :param x:
        :param y:
        :param train_size:
        :param random_state:
        :param stratify:
        :return:
        """

        return model_selection.train_test_split(x, y, train_size=train_size,
                                                random_state=random_state, stratify=stratify)

    def for_generator(self, x: pd.DataFrame, y: pd.DataFrame, labels: list, group: str) -> pd.DataFrame:
        """
        Preparing a metadata set for a generator that creates the images delivery pipeline

        :param x: The features part of a data set
        :param y: The corresponding classes/labels
        :param labels: The list of the labels
        :param group: The data group in focus, i.e., training, validating, or testing; for logging purposes.
        :return:
        """

        data = x[['url']].join(y[labels], how='inner')
        self.logger.info("{}: {}, {}, {}".format(group, x.shape, y.shape, data.shape))
        return data

    def summaries(self, data: pd.DataFrame, features: list, labels: list) -> (pd.DataFrame,
                                                                              pd.DataFrame, pd.DataFrame):
        """
        Performs stratified data splitting

        :param data: The data set to be split
        :param features: The names of the feature columns
        :param labels: The names of the label columns
        :return:
        """

        # Stratified Splitting
        x_learn, x_evaluation, y_learn, y_evaluation = \
            self.stratification(x=data[features], y=data[labels], train_size=self.train_size_initial,
                                random_state=self.random_state, stratify=data[labels])

        x_validate, x_test, y_validate, y_test = \
            self.stratification(x=x_evaluation, y=y_evaluation, train_size=self.train_size_evaluation,
                                random_state=self.random_state, stratify=y_evaluation)

        # Setting-up for generator
        training = self.for_generator(x=x_learn, y=y_learn, labels=labels, group='Training')
        validating = self.for_generator(x=x_validate, y=y_validate, labels=labels, group='Validating')
        testing = self.for_generator(x=x_test, y=y_test, labels=labels, group='Testing')

        return training, validating, testing
