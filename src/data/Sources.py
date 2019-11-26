import pandas as pd
import configurations.configurations as cfg
import logging
import sys


class Sources:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.disabled = True

    # The data classifications
    @staticmethod
    def truth():
        try:
            truth = pd.read_csv(cfg.variables()['data']['source']['truth'])
        except Exception as e:
            print(e)
            sys.exit(1)

        return truth, truth.shape[0]

    @staticmethod
    def metadata():
        try:
            metadata = pd.read_csv(cfg.variables()['data']['source']['metadata'])
        except Exception as e:
            print(e)
            sys.exit(1)

        return metadata, metadata.shape[0]

    @staticmethod
    def summary():
        truth, _ = Sources.truth()
        metadata, _ = Sources.metadata()

        # Hence
        listing = metadata.merge(truth, on='image', how='inner')

        # The truth labels are one-hot-coded.  The labels are
        # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'
        labels = truth.columns.drop('image').values.tolist()

        # Fields
        fields = listing.columns.drop(labels).values.tolist()

        # Hence
        assert listing[labels].sum(axis=1).all(), "Each image must be associated with a single class only"
        return listing, labels, fields
