import logging

import sklearn.model_selection as model_selection

import configurations.configurations as cfg
import src.data.images as images
import src.data.usable as usable


def main():
    # Logging
    cfg.logs()
    logger = logging.getLogger('root')
    logger.name = __name__

    # Preliminaries
    listing, labels, fields = usable.Usable().summary()

    # Split
    random_state = cfg.variables()['modelling']['parameters']['random_state']
    xlearn, xtest, ylearn, ytest = model_selection \
        .train_test_split(listing.drop(columns=labels).values, listing[labels],
                          train_size=0.7, random_state=random_state, stratify=listing[labels])

    x = images.Images().states(listing['image'])
    logger.info(x)


if __name__ == '__main__':
    main()
