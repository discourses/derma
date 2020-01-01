import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import src.data.source as source
    import src.data.transform as transform
    import src.data.graphing as graphing

    import src.modelling.VGG19E.simple as vgg19e_simple
    import src.modelling.VGG19E.architecture as vgg19e_arc


def main():

    # Data
    inventory, labels, features = source.Source().summaries()
    training, validating, testing = transform.Transform().summaries(inventory, features, labels)

    # Graphing
    graphs = graphing.Graphing()

    # VGG19 Extract
    arc = vgg19e_arc.Architecture()
    hyperparameters = arc.hyperparameters()

    # Experiment
    vgg = vgg19e_simple.Simple(training_=training, validating_=validating, testing_=testing,
                               labels=labels, epochs=2)
    print(hyperparameters[0])
    history = vgg.run(hyperparameters=hyperparameters[0])

    print(type(history))
    print(history.history['val_auc'])

    graphs.plot_metrics(history=history)


if __name__ == '__main__':
    main()
