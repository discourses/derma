import logging
import os

import src.data.Sources as Sources
import src.data.DermoscopicImages as DermoscopicImages

import sklearn.model_selection as model_selection


def main():

    print(os.getcwd())

    # Logging, logging.disable(logging.WARN)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Preliminaries
    random_state = 5
    listing, labels, features = Sources.Sources().summary()
    xlearn, xtest, ylearn, ytest = model_selection\
        .train_test_split(listing.drop(columns=labels).values, listing[labels],
                          train_size=0.7, random_state=random_state, stratify=listing[labels])

    listing.head()
    print(listing['image'])

    states = DermoscopicImages.DermoscopicImages().states(listing['image'])
    print(states.status.all())


if __name__ == '__main__':
    main()
