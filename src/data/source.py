"""Module source"""
import glob
import inspect
import io
import os
import sys

import pandas as pd
import requests

import config
import src.data.sampling as sampling


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

        # Variables
        variables = config.Config().variables()

        # Inventory
        self.inventory_url = variables['inventory']['url']
        self.inventory_fields = variables['inventory']['fields']

        # Images
        self.images_path = variables['images']['path']
        self.images_ext = variables['images']['ext']

        # Modelling
        self.features = variables['modelling']['features']
        self.sample = variables['modelling']['sample']

    def inventory(self) -> pd.DataFrame:
        """
        Downloads the metadata summary of images.

        :return:
            Metadata DataFrame
        """

        # Examine the name of function that called this function
        caller = inspect.stack()[1]
        testing = (caller.function == 'test_inventory') | (caller.function == 'test_url')

        # Download the inventory of the images metadata
        try:
            req = requests.get(self.inventory_url)
            inventory = pd.read_csv(io.StringIO(req.content.decode(encoding='utf-8')))
        except OSError as error:
            print(error)
            sys.exit(1)

        assert req.status_code == 200, "Not 200"

        # If the calling function is not a test function, cf. inventory & downloaded images
        if not testing:
            imports = glob.glob(os.path.join(self.images_path, '*{}'.format(self.images_ext)))
            accessible = pd.DataFrame(imports, columns=['name'])
            accessible['name'] = accessible.name.apply(lambda x: os.path.split(x)[1])
            inventory = accessible.merge(inventory, how='inner', on='name')

        return inventory

    def url(self, inventory: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the field 'url' to the inputted data frame.  It records the path - including the image name - to an image.

        :type inventory: pandas.DataFrame

        :param inventory:
        :return:
            pandas DataFrame inventory
        """

        inventory['url'] = inventory.name.apply(lambda x: os.path.join(self.images_path, x))

        return inventory

    def summaries(self):
        """
        Uploads the inventory of images

        :return:
        """

        # Read-in the inventory of images
        inventory = Source().inventory()
        fields = self.inventory_fields
        labels = inventory.columns.drop(fields).tolist()

        # Sample
        if self.sample:
            inventory = sampling.Sampling().sample(data=inventory, fields=fields, labels=labels)

        # Add the images URL field
        inventory = Source().url(inventory)

        # Return
        features = self.features
        return inventory, labels, features
