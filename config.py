import pandas as pd


# Global Variables
# with open('https://raw.githubusercontent.com/greyhypotheses/derma/develop/config.json') as file:
#     raw = json.load(file)
#
# variables = json.loads(json.dumps(raw))

variables = pd.read_json('https://raw.githubusercontent.com/greyhypotheses/derma/develop/config.json', 'records')
