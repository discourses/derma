import pandas as pd
import logging
import config

# try proposed via new class

# def truth
# hence test_truth

# def metadata
# hence test_metadata

# def inventory
# hence test_inventory
# logic w.r.t. length truth, length metadata, length dataset
# labels not empty, features not empty


def inventory():

    # Logging, logging.disable(logging.WARN)
    logging.basicConfig(level=logging.DEBUG)

    # For debugging purposes
    logger = logging.getLogger(__name__)

    # The data classifications
    truth = pd.read_csv(config.variables['data']['source']['truth'])
    logger.info(truth.head())

    # The metadata of the data/images
    metadata = pd.read_csv(config.variables['data']['source']['metadata'])
    logger.info(metadata.head())

    # Hence
    dataset = metadata.merge(truth, on='image', how='inner')
    logger.info(f"The length of the dataset: {len(dataset)}")
    logger.info(dataset.head())

    # The truth labels are one-hot-coded.  The labels are
    # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'
    labels = truth.columns.drop('image').values.tolist()
    logger.info(labels)

    # Features
    features = dataset.columns.drop(labels).values.tolist()
    logger.info(features)

    # Hence
    return dataset, labels, features
