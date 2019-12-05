import sys

import pandas as pd
import requests

import configurations.configurations as cfg


# Due to the size of the ISIC data set, an initial consideration was
# to extract images, in time, directly from the ISIC archive.  For such an approach
# the identification code of each image is required, hence this method.  This approach
# has been abandoned due to the risks.
class Archive:
    """
    The Archive object
    """

    def __init__(self):
        """
        The constructor
        """
        self.name = 'Archive'

    @staticmethod
    def images():

        # URL
        version = cfg.variables()['api']['isic']['version']
        retrieve = 'image'
        url = f"{cfg.variables()['api']['isic']['url']}{version}/{retrieve}"

        # Parameters
        limit = '30000'
        sort = 'name'
        sortdir = '1'
        detail = 'false'
        parameters = {'limit': limit, 'sort': sort, 'sortdir': sortdir, 'detail': detail}

        # Request
        # API Structure: https://isic-archive.com/api/v1/image?limit=50&sort=name&sortdir=1&detail=false
        try:
            r = requests.get(url, params=parameters)
        except requests.exceptions.RequestException as e:
            print(e)
            sys.exit(1)

        # The data frame of the images and their identification codes
        summary = pd.DataFrame.from_dict(r.json())
        summary.rename(columns={'_id': 'id'}, inplace=True)

        # A data frame consisting of image identification code (id),
        # image name (name), and last update time stamp (updated) is returned.
        return summary
