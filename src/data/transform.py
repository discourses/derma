import logging
import os

import pandas as pd
import sklearn.model_selection as model_selection

import src.federal.federal as federal


class Transform:

    def __init__(self):
        # Root directory
        self.root = os.path.split(os.getcwd())[0]

        # Logging
        federal.Federal().logs()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('root')
        self.logger.name = __name__

        # Variables
        variables = federal.Federal().variables()
        self.random_state = variables['modelling']['random_state']
        self.train_size_inventory = variables['modelling']['train_size_inventory']
        self.train_size_evaluation = variables['modelling']['train_size_evaluation']


    @staticmethod
    def stratification(x, y, train_size, random_state, stratify):
        return model_selection.train_test_split(x, y,
                                                train_size=train_size,
                                                random_state=random_state, stratify=stratify)


    def for_generator(self, x, y, labels, group):
        """
        :type x: numpy.ndarray
        :type y: numpy.ndarray
        :type labels: list
        :type group: str

        :param x: The features part of a data set
        :param y: The corresponding classes/labels
        :param labels: The list of the labels
        :param group: The data group in focus, i.e., training, validating, or testing; for logging purposes.
        :return:
        """

        data = pd.DataFrame(x, columns=['url']) \
            .join(pd.DataFrame(y, columns=labels), how='inner')

        self.logger.info("{}: {}, {}, {}".format(group, x.shape, y.shape, data.shape))

        return data


    def summaries(self, data, features, labels):

        # Stratified Splitting
        x_learn, x_evaluation, y_learn, y_evaluation = \
            Transform().stratification(x=data[features], y=data[labels], train_size=self.train_size_inventory,
                                       random_state=self.random_state, stratify=data[labels])

        x_validate, x_test, y_validate, y_test = \
            Transform().stratification(x=x_evaluation, y=y_evaluation, train_size=self.train_size_evaluation,
                                       random_state=self.random_state, stratify=y_evaluation)

        # Setting-up for generator
        training = Transform().for_generator(x=x_learn, y=y_learn, labels=labels, group='Training')
        validating = Transform().for_generator(x=x_validate, y=y_validate, labels=labels, group='Validating')
        testing = Transform().for_generator(x=x_test, y=y_test, labels=labels, group='Testing')

        return training, validating, testing
