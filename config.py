import os
import sys
import typing
import datetime

import requests
import yaml


class Config:
    """
    Consists of methods that parse the global settings summarised in the online
    YAML dictionaries
    """

    def __init__(self):
        """
        The constructor
        """
        self.root = os.path.abspath(__package__)

    def paths(self, partitions: list):
        """
        Creates a path relative to the project's root directory
        :param partitions:
        :return:
            path: The created from a list of directories
        """

        path = self.root
        for partition in partitions:
            path = os.path.join(path, partition)

        return path

    def variables(self) -> typing.Dict:
        """
        Parses the global variables file of this project

        :return:
        """

        url = 'https://raw.githubusercontent.com/greyhypotheses/dictionaries/develop/derma/variables.yml'
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)
        variables = yaml.safe_load(req.text)

        # Image Dimension
        variables['images']['image_dimension'] = (variables['images']['rows'], variables['images']['columns'],
                                                  variables['images']['channels'])

        # local Storage
        variables['modelling']['model_checkpoints_path'] = \
            self.paths(variables['modelling']['model_checkpoints_directory'])
        variables['images']['path'] = self.paths(variables['zipped']['images']['unzipped'])

        # Cloud Storage
        datetime_string = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
        variables['modelling']['s3_path'] = variables['modelling']['s3_bucket'] + datetime_string + '/'

        return variables

    @staticmethod
    def logs() -> typing.Dict:
        """
        Parses the logs settings file of this project

        :return:
        """

        url = 'https://raw.githubusercontent.com/greyhypotheses/dictionaries/develop/derma/logs.yml'
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)
        logs = yaml.safe_load(req.text)

        return logs
