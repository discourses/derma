"""Module source"""
import os
import sys
import typing

import pandas as pd

import src.config as config


class Source:
    """
    Class Source

    The methods herein download and prepare the metadata of the images that would be used for modelling
    """


    def __init__(self):
        """
        The constructor

        Herein, the constructor initialises the global variables used by the methods of this class.
        """

        # Root directory
        self.root = os.path.split(os.getcwd())[0]

        # Variables
        variables = config.Config().variables()

        # Inventory
        self.inventory_url = variables['inventory']['url']
        self.inventory_fields = variables['inventory']['fields']

        # Images
        self.images_location = variables['images']['location']
        self.images_ext = variables['images']['ext']

        # Modelling
        self.features = variables['modelling']['features']
        self.random_state = variables['modelling']['random_state']
        self.sample = variables['modelling']['sample']
        self.replace = variables['modelling']['replace']
        self.class_sample_size = variables['modelling']['class_sample_size']


    def inventory(self) -> pd.DataFrame:
        """
        Downloads the metadata summary of images.

        :return:
            Metadata DataFrame
        """

        try:
            inventory = pd.read_csv(self.inventory_url)
        except OSError as error:
            print(error)
            sys.exit(1)

        inventory['name'] = inventory['image'] + inventory.angle.apply(
            lambda x: '-' + str(x).zfill(3) + self.images_ext)

        return inventory


    def url(self, inventory: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the field 'url' to the inputted data frame.  It records the path - including the image name - to an image.

        :type inventory: pandas.DataFrame

        :param inventory:
        :return:
            pandas DataFrame inventory
        """

        images_location = self.root
        for i in self.images_location:
            images_location = os.path.join(images_location, i)

        inventory['url'] = inventory.name.apply(lambda x: os.path.join(images_location, x))

        return inventory


    def sampling(self, data: pd.DataFrame, fields: typing.List, labels: typing.List) -> pd.DataFrame:
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


    def summaries(self):

        # Read-in the inventory of images
        inventory = Source().inventory()
        fields = self.inventory_fields
        labels = inventory.columns.drop(fields).tolist()

        # Sample
        if self.sample:
            inventory = Source().sampling(data=inventory, fields=fields, labels=labels)

        # Add the images URL field
        inventory = Source().url(inventory)

        # Return
        features = self.features
        return inventory, labels, features
