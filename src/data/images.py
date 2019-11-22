import requests
import pandas as pd
import multiprocessing as mp
import config
import logging


def state(image):
    r = requests.get(config.variables['data']['source']['images'] + image + '.jpg')
    return {'image': image, 'status': r.status_code}


def states(images):

    # Logging, logging.disable(logging.WARN)
    logging.basicConfig(filename='states.txt', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Parallel Processing via CPU
    pool = mp.Pool(mp.cpu_count())

    # For a small data frame of image names
    excerpt = images.sample(n=16)

    # An iterable form of excerpt
    listing = [{excerpt.image[i]} for i in excerpt.index]

    # Parallel processing of states
    sample = [pool.apply(state, args=i) for i in listing]

    # Log states of ...
    logger.info(sample)

    # The access states of images
    return pd.DataFrame(sample)
