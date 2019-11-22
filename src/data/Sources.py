import pandas as pd
import config
import logging


class Sources:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    # The data classifications
    @staticmethod
    def truth():
        truth = pd.read_csv(config.variables['data']['source']['truth'])
        return truth, truth.shape[0]

    @staticmethod
    def metadata():
        metadata = pd.read_csv(config.variables['data']['source']['metadata'])
        return metadata, metadata.shape[0]

    def summary(self):
        truth, _ = Sources.truth()
        metadata, _ = Sources.metadata()

        # Hence
        listing = metadata.merge(truth, on='image', how='inner')
        self.logger.info(listing.head())

        # The truth labels are one-hot-coded.  The labels are
        # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'
        labels = truth.columns.drop('image').values.tolist()

        # Features
        features = listing.columns.drop(labels).values.tolist()

        # Hence
        return listing, labels, features
