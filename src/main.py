import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import src.data.source as source
    import src.data.transform as transform
    import src.config as config

    import src.modelling.VGG19FE.simple as vgg19e_simple
    import src.modelling.VGG19FE.architecture as vgg19e_arc
    import src.modelling.VGG19FE.hyperparameters as vgg19e_hyp


def main():

    # Data
    inventory, labels, features = source.Source().summaries()
    training, validating, testing = transform.Transform().summaries(inventory, features, labels)
    print(training.head())

    variables = config.Config().variables()
    print(variables['images']['image_dimension'])

    # VGG19 Extract
    # arc = vgg19e_arc.Architecture()
    # hyp = vgg19e_hyp.Hyperparameters()

    # Experiment
    # vgg = vgg19e_simple.Simple(training_=training, validating_=validating, testing_=testing,
    #                            labels=labels, epochs=2)
    # print(hyp.values()[0])
    # history = vgg.run(hyperparameters=hyp.values()[0])


if __name__ == '__main__':
    main()
