import os

import yaml


class Cfg:

    def __init__(self):
        self.path = os.path.split(os.path.abspath(__file__))[0]

    def variables(self):
        """
        Parses the generic variables file of this project

        :return:
        """

        with open(os.path.join(self.path, 'variables.yml'), 'r') as file:
            variables = yaml.safe_load(file)

        variables['images']['image_dimension'] = (variables['images']['rows'], variables['images']['columns'],
                                                  variables['images']['channels'])

        return variables

    def logs(self):
        """
        Parses the logs settings file of this project

        :return:
        """

        with open(os.path.join(self.path, 'logs.yml'), 'r') as file:
            logs = yaml.safe_load(file)

        return logs
