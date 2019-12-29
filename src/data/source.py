import logging
import os

import pandas as pd
import sklearn.model_selection as model_selection

from src.cfg import cfg as cfg


class Source:

    def __init__(self):

        # Root directory
        self.root = os.path.split(os.getcwd())[0]

        # Logging
        cfg.Cfg().logs()
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('root')
        self.logger.name = __name__

        # Variables
        variables = cfg.Cfg().variables()
        self.inventory_filename = variables['inventory']['filename']
        self.images_location = variables['images']['location']
        self.images_ext = variables['images']['ext']
        self.inventory_fields = variables['inventory']['fields']
        self.random_state = variables['modelling']['random_state']
        self.train_size_inventory = variables['modelling']['train_size_inventory']
        self.train_size_evaluation = variables['modelling']['train_size_evaluation']


    def inventory(self):

        inventory_filename = self.root
        for i in self.inventory_filename:
            inventory_filename = os.path.join(inventory_filename, i)

        return pd.read_csv(inventory_filename)


    def url(self, inventory):

        images_location = self.root
        for i in self.images_location:
            images_location = os.path.join(images_location, i)

        inventory['name'] = inventory['image'] + inventory.angle.apply(
            lambda x: '-' + str(x).zfill(3) + self.images_ext)
        inventory['url'] = inventory.name.apply(lambda x: os.path.join(images_location, x))

        return inventory


    def builder(self, x, y, labels, group):
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


    def summaries(self):

        # Read-in the inventory of images
        inventory = Source().inventory()
        labels = inventory.columns.drop(self.inventory_fields)

        # Add the images URL field
        inventory = Source().url(inventory)

        # Stratified Splitting
        x_learn, x_evaluation, y_learn, y_evaluation = model_selection. \
            train_test_split(inventory[['url']], inventory[labels],
                             train_size=self.train_size_inventory,
                             random_state=self.random_state, stratify=inventory[labels])

        x_validate, x_test, y_validate, y_test = model_selection. \
            train_test_split(x_evaluation, y_evaluation,
                             train_size=self.train_size_evaluation,
                             random_state=self.random_state, stratify=y_evaluation)

        # Setting-up for generator
        training = Source().builder(x=x_learn, y=y_learn, labels=labels, group='Training')
        validating = Source().builder(x=x_validate, y=y_validate, labels=labels, group='Validating')
        testing = Source().builder(x=x_test, y=y_test, labels=labels, group='Testing')

        return training, validating, testing, labels
