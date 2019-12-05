import logging
import multiprocessing as mp

import pandas as pd
import requests

import configurations.configurations as cfg


class Images:
    """
    The Images object
    """

    def __init__(self):
        """
        The constructor
        """

        # Logging
        cfg.logs()
        self.logger = logging.getLogger('debug')
        self.logger.name = __name__

    @staticmethod
    def state(image):
        """

        :param image:
        :return:
        """

        # Image
        r = requests.get(cfg.variables()['data']['source']['images'] + image + '.jpg')

        # Return image & access status code
        return {'image': image, 'status': 1 if r.status_code == 200 else 0}

    def states(self, images):
        """

        :param images:
        :return:
        """

        # Parallel Processing via CPU
        pool = mp.Pool(mp.cpu_count())

        # For a small data frame of image names
        # images.sample(n=config.variables['tests']['random_sample_size']['images'])
        excerpt = images.sample(n=cfg.variables()['tests']['random_sample_size']['images'])

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
