import math
import tensorflow as tf

import src.cfg.cfg as cfg
import src.data.graphing as graphing
import src.data.reader as reader
import src.modelling.VGG19E.architecture as arc
import src.modelling.metrics as met


class Simple:

    def __init__(self, training_, validating_, testing_, labels, epochs):
        """

        :param training_:
        :param validating_:
        :param testing_:
        :param labels:
        :param epochs:
        """

        # Variables
        variables = cfg.Cfg().variables()
        self.batch_size = variables['modelling']['batch_size']

        # Data
        read = reader.Reader()
        self.training = read.generator(training_, labels)
        self.validating = read.generator(validating_, labels)
        self.testing = read.generator(testing_, labels)

        # The labels/classes
        self.labels = labels

        # Number of epochs
        self.epochs = epochs

        # Logs
        self.base = 'logs'

    def instance(self, model):
        """

        :param model:
        :return:
        """

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True
        )

        history = model.fit_generator(generator=self.training,
                                      steps_per_epoch=math.floor(len(self.training) / self.batch_size),
                                      epochs=self.epochs,
                                      verbose=1,
                                      callbacks=[early_stopping],
                                      validation_data=self.validating,
                                      validation_steps=math.floor(len(self.validating) / self.batch_size))
        return history

    def run(self):
        """

        :return:
        """

        # Metrics
        metrics = met.Metrics()

        # Architecture
        architecture = arc.Architecture()
        hyper = architecture.hyper()

        # Hyperparameters combinations enumerator
        combination = 0

        graphs = graphing.Graphing()

        for hyp in hyper:
            model: tf.keras.preprocessing.image.ImageDataGenerator = \
                architecture.core(hyp=hyp, metrics=metrics.classification(), labels=self.labels)

            history = Simple.instance(self, model=model)

            graphs.plot_metrics(history=history)

            combination += 1
