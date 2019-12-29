import src.data.source as source

import src.modelling.VGG19E.complex as vgg19e


def main():

    # Data
    training, validating, testing, labels = source.Source().summaries()

    # VGG19 Extract
    vgg = vgg19e.Steps(training_=training, validating_=validating, testing_=testing,
                       labels=labels, epochs=2)
    vgg.run()


if __name__ == '__main__':
    main()
