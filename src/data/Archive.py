import requests
import pandas as pd
import sys


# Due to the size of the ISIC data set, an initial consideration was
# to extract images, as needed, directly from the ISIC archive.  For such an approach
# the identification code of each image is required, hence this method.
#
# However, the approach has been abandoned due to the risks.
#
# API Structure: https://isic-archive.com/api/v1/image?limit=50&sort=name&sortdir=1&detail=false
class Archive:

    def __init__(self):

        # URL
        self.version = 'v1'
        self.retrieve = 'image'
        self.url = f"https://isic-archive.com/api/{self.version}/{self.retrieve}"

        # Parameters
        self.limit = '30000'
        self.sort = 'name'
        self.sortdir = '1'
        self.detail = 'false'
        self.parameters = {'limit': self.limit, 'sort': self.sort, 'sortdir': self.sortdir, 'detail': self.detail}

    def images(self):

        # Request
        try:
            r = requests.get(self.url, params=self.parameters)
        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)

        # The data frame of the images and their identification codes
        summary = pd.DataFrame.from_dict(r.json())
        summary.rename(columns={'_id': 'id'}, inplace=True)

        # A data frame consisting of image identification code (id),
        # image name (name), and last update time stamp (updated) is returned.
        return summary
