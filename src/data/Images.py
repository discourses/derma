import logging
import requests
import multiprocessing as mp
import pandas as pd

import config


class Images:

    def __init__(self):
        logging.basicConfig(filename='Images.txt', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def state(image):
        r = requests.get(config.variables['data']['source']['images'] + image + '.jpg')
        return {'image': image, 'status': True if r.status_code == 200 else False}

    def states(self, images):

        # Parallel Processing via CPU
        pool = mp.Pool(mp.cpu_count())

        # For a small data frame of image names
        # images.sample(n=config.variables['tests']['random_sample_size']['images'])
        excerpt = images.sample(n=config.variables['tests']['random_sample_size']['images'])

        # An iterable form of excerpt
        listing = [{excerpt[i]} for i in excerpt.index]

        # Parallel processing of states
        # sample = [pool.apply(Images.state, args=i) for i in listing]
        sample = pool.starmap_async(Images.state, [i for i in listing]).get()

        # Log states of ...
        self.logger.info(sample)

        # The access states of images
        return pd.DataFrame(sample)
