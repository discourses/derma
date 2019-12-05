import sys

import pandas as pd

import configurations.configurations as cfg


class Sources:
    """
    Processes source data
    """

    def __init__(self):
        self.name = 'Sources'

    # The data classifications
    @staticmethod
    def truth():
        """
        Reads the ground truth data file.  The location of the file is recorded
        in the 'configurations/variables.json' file
        :return:
            truth: A data frame of the data in the file
            truth.shape[0]: The number of rows/instances in the data frame
        """
        try:
            truth = pd.read_csv(cfg.variables()['data']['source']['truth'])
        except Exception as e:
            print(e)
            sys.exit(1)

        return truth, truth.shape[0]

    @staticmethod
    def metadata():
        """
        Reads the metadata data file.  The location of the file is recorded
        in the 'configurations/variables.json' file
        :return:
            metadata: A data frame of the data in the file
            metadata.shape[0]: The number of rows/instances in the data frame
        """
        try:
            metadata = pd.read_csv(cfg.variables()['data']['source']['metadata'])
        except Exception as e:
            print(e)
            sys.exit(1)

        return metadata, metadata.shape[0]

    @staticmethod
    def summary():
        """
        Reads the 'truth' & 'metadata' files and joins their data via the image name field 'image'
        :return:
            listing: The joined 'truth' & 'metadata' data
            labels: The list of the data's classes; the classes are one-hot-coded.
            fields: The list of metadata fields, excluding labels.
        """
        truth, _ = Sources.truth()
        metadata, _ = Sources.metadata()

        # Hence
        # Join the metadata & truth data frames via common field 'image'
        listing = metadata.merge(truth, on='image', how='inner')
        listing.drop_duplicates(keep='first', inplace=True)

        # The truth labels are one-hot-coded.  The labels are
        # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'
        labels = truth.columns.drop('image').values.tolist()

        # The metadata fields
        fields = listing.columns.drop(labels).values.tolist()

        # Herein,
        #   the class/label fields are selected: listing[labels]
        #   then the sum-per-row is calculated: listing[labels].sum(axis=1)
        # If each image is associated with a single class then the sum of each row will be 1, i.e.,
        #   listing[labels].sum(axis=1).all() will be True
        assert listing[labels].sum(axis=1).all(), "Each image must be associated with a single class only"

        # Hence
        return listing, labels, fields
