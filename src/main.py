import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import src.data.source as source
    import src.data.transform as transform

    import src.modelling.extraction as extraction


def main():
    # Data
    inventory, labels, features = source.Source().summaries()
    training_, validating_, testing_ = transform.Transform().summaries(inventory, features, labels)

    # Steps w.r.t. a feature extraction transfer learning model
    extraction.steps.Steps().proceed(labels=labels,
                                     epochs=2,
                                     training_=training_,
                                     validating_=validating_,
                                     testing_=testing_)


if __name__ == '__main__':
    main()
