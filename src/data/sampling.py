import typing

import pandas as pd

import src.config as config


class Sampling:

    def __init__(self):

        self.name = 'Sampling'

        # Variables
        variables = config.Config().variables()

        # Modelling
        self.random_state = variables['modelling']['random_state']
        self.replace = variables['modelling']['replace']
        self.class_sample_size = variables['modelling']['class_sample_size']


    def sample(self, data: pd.DataFrame, fields: typing.List, labels: typing.List) -> pd.DataFrame:
        """
        Provides a sample of the data

        :type data: pandas.DataFrame
        :type fields: List
        :type labels: List

        :param data: A data frame of the data set wherefrom a sample is extract
        :param fields: The fields of the data set, excluding the class/label columns
        :param labels: The labels of the data set
        :return:
            A sample of data
        """

        # Counts the number of records per class/label
        n_per_label = data[labels].sum(axis=0)

        # If sampling without replacement is required, but one or more classes have fewer records than
        # the requested sample size, the sample size is assigned as outlined below
        if (not self.replace) & (n_per_label.min() < self.class_sample_size):
            print("Because sampling without replacement has been requested, and to ensure a balanced "
                  "data set, the sample size has been changed to the size of the smallest class")
            class_sample_size = n_per_label.min()
        else:
            class_sample_size = self.class_sample_size

        # Hence
        excerpt = data.groupby(labels)[fields + labels] \
            .apply(lambda x: x.sample(n=class_sample_size, replace=self.replace, random_state=self.random_state))
        excerpt.reset_index(drop=True, inplace=True)

        return excerpt
