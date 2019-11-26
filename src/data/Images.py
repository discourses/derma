import logging
import requests
import multiprocessing as mp
import pandas as pd

import config


class Images:

    def __init__(self):

        # Proceed from
        # logging.config.dictConfig(dictionary)

        # Logging: Temporary Approach
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        handler = logging.FileHandler(filename='images.log')
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    @staticmethod
    def state(image):
        r = requests.get(config.variables['data']['source']['images'] + image + '.jpg')
        return {'image': image, 'status': 1 if r.status_code == 200 else 0}

    def states(self, images):

        # Parallel Processing via CPU
        pool = mp.Pool(mp.cpu_count())

        # For a small data frame of image names
        # images.sample(n=config.variables['tests']['random_sample_size']['images'])
        excerpt = images.sample(n=config.variables['tests']['random_sample_size']['images'])

        # An iterable form of excerpt
        excerpt_iterable = [{excerpt[i]} for i in excerpt.index]

        # Determine whether each of the randomly selected images exists in the repository
        # image, status
        sample_dict = pool.starmap_async(Images.state, [i for i in excerpt_iterable]).get()
        sample_frame = pd.DataFrame(sample_dict)

        # Are there any missing images?
        missing_images = sample_frame[sample_frame.status == 0]
        self.logger.info(missing_images)

        # The status codes of the images
        return sample_frame
