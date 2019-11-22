import logging
import os

import src.data.inventory as inventory

import sklearn.model_selection as model_selection


def main():

    print(os.getcwd())

    # Logging, logging.disable(logging.WARN)
    logging.basicConfig(level=logging.DEBUG)

    # Preliminaries
    random_state = 5
    dataset, labels, features = inventory.inventory()
    xlearn, xtest, ylearn, ytest = model_selection\
        .train_test_split(dataset.drop(columns=labels).values, dataset[labels],
                          train_size=0.7, random_state=random_state, stratify=dataset[labels])

    dataset.head()


if __name__ == '__main__':
    main()
