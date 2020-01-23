import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import src.data.source as source
    import src.data.transform as transform

    import src.modelling.extraction.steps as fe
    import config


def main():
    variables = config.Config().variables()

    # Reading-in a metadata table of the images, and lists summarising the table's label columns & feature columns
    inventory, labels, features = source.Source().summaries()

    # Splitting the data into training, validating, and testing sets.  The sets are image metadata tables.
    training_, validating_, testing_ = transform.Transform().summaries(inventory, features, labels)

    # Model: feature extraction transfer learning model
    fe.Steps().proceed(labels=labels,
                       epochs=variables['modelling']['epochs'],
                       training_=training_,
                       validating_=validating_,
                       testing_=testing_)


if __name__ == '__main__':
    main()
