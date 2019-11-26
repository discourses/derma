import json
import logging.config
import os
import pandas as pd


lead = os.path.split(os.path.abspath(__file__))[0]


def variables():
    return pd.read_json(os.path.join(lead, 'variables.json'), 'records')


def logs():
    with open(os.path.join(lead, 'logs.json'), 'r') as filename:
        dictionary = json.load(filename)

    return logging.config.dictConfig(dictionary)
