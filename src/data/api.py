import requests
import pandas as pd
import sys


# Due to the size of the ISIC data set, an initial consideration was
# to extract images, as needed, directly from the ISIC archive.  For such an approach
# the identification code of each image is required, hence this method.
#
# However, the approach has been abandoned due to the risks.
def api():

    # API Structure
    # https://isic-archive.com/api/v1/image?limit=50&sort=name&sortdir=1&detail=false

    # URL
    version = 'v1'
    retrieve = 'image'
    url = f"https://isic-archive.com/api/{version}/{retrieve}"

    # Parameters
    parameters = {'limit': '30000', 'sort': 'name', 'sortdir': '1', 'detail': 'false'}

    # Hence, request
    try:
        r = requests.get(url, params=parameters)
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)

    # The data frame of the images and their identification codes
    identifiers = pd.DataFrame.from_dict(r.json())
    identifiers.rename(columns={'_id': 'id'}, inplace=True)

    # A data frame consisting of image identification code (id),
    # image name (name), and last update time stamp (updated) is returned.
    return identifiers
