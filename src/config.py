import typing
import os
import sys
import yaml
import requests


class Config:

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

    @staticmethod
    def variables() -> typing.Dict:
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

        variables['images']['image_dimension'] = (variables['images']['rows'], variables['images']['columns'],
                                                  variables['images']['channels'])

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
