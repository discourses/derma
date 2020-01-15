"""Module federal"""
import os

import yaml

import typing


class Federal:
    """
    Class Federal
    Consists of methods that parse the global settings summarised in the
    YAML dictionaries of directory federal
    """

    def __init__(self):
        """
        The constructor
        """
        self.path = os.path.split(os.path.abspath(__file__))[0]

    def variables(self) -> typing.Dict:
        """
        Parses the global variables file of this project

        :return:
        """

        with open(os.path.join(self.path, 'variables.yml'), 'r') as file:
            variables = yaml.safe_load(file)

        variables['images']['image_dimension'] = (variables['images']['rows'], variables['images']['columns'],
                                                  variables['images']['channels'])

        return variables

    def logs(self) -> typing.Dict:
        """
        Parses the logs settings file of this project

        :return:
        """

        with open(os.path.join(self.path, 'logs.yml'), 'r') as file:
            logs = yaml.safe_load(file)

        return logs
