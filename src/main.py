import logging
import configurations.configurations as cfg

import src.data.Usable as Usable
import src.data.Images as Images

import sklearn.model_selection as model_selection


def main():

    # Logging
    cfg.logs()
    logger = logging.getLogger(__name__)

    # Preliminaries
    listing, labels, fields = Usable.Usable().records()

    # Split
    random_state = cfg.variables()['modelling']['parameters']['random_state']
    xlearn, xtest, ylearn, ytest = model_selection\
        .train_test_split(listing.drop(columns=labels).values, listing[labels],
                          train_size=0.7, random_state=random_state, stratify=listing[labels])
    logger.info(xlearn[0])

    x = Images.Images().states(listing['image'])
    logger.info(x)


if __name__ == '__main__':
    main()
