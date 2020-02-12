import os
import sys
import argparse
import requests
import yaml


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import src.data.source as source
    import src.data.transform as transform

    import src.modelling.extraction.steps as extraction
    import config


def main():
    # The hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("hyperparameters_url")
    args = parser.parse_args()

    try:
        req = requests.get(args.hyperparameters_url)
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)
    hyperparameters_dict = yaml.safe_load(req.text)

    print("\nThe hyperparameters file: ")
    print(args.hyperparameters_url)
    print("\nThe hyperparameters: ")
    print(hyperparameters_dict)

    # The global variables
    variables = config.Config().variables()

    # Reading-in a metadata table of the images, and lists summarising the table's label columns & feature columns
    # inventory: DataFrame
    # labels: list
    # features: list
    inventory, labels, features = source.Source().summaries()

    # Splitting the data into training, validating, and testing sets.  The sets are image metadata tables.
    # training_: DataFrame
    # validating_: DataFrame
    # testing_: DataFrame
    training_, validating_, testing_ = transform.Transform().summaries(inventory, features, labels)

    # Model: feature extraction transfer learning model
    extraction.Steps().evaluate(labels=labels,
                                epochs=variables['modelling']['epochs'],
                                hyperparameters_dict=hyperparameters_dict,
                                training_=training_,
                                validating_=validating_,
                                testing_=testing_)


if __name__ == '__main__':
    main()
