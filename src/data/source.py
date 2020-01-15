"""Module source"""
import os
import sys
import pandas as pd

from src.federal import federal as federal


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
        variables = federal.Federal().variables()

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
        self.class_sample_size = variables['modelling']['class_sample_size']


    def inventory(self) -> pd.DataFrame:
        """
        Downloads the metadata summary of images
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


    def url(self, inventory):

        images_location = self.root
        for i in self.images_location:
            images_location = os.path.join(images_location, i)

        inventory['url'] = inventory.name.apply(lambda x: os.path.join(images_location, x))

        return inventory


    def sampling(self, data, fields, labels):

        excerpt = data.groupby(labels)[fields + labels] \
            .apply(lambda x: x.sample(n=self.class_sample_size, replace=False, random_state=self.random_state))
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
