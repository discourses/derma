import math
import tensorflow as tf

import src.cfg.cfg as cfg
import src.data.reader as reader
import src.modelling.VGG19E.architecture as arc
import src.evaluating.metrics as met


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

        self.n_training = training_.shape[0]
        self.n_validating = validating_.shape[0]
        self.n_testing = testing_.shape[0]

        # The labels/classes
        self.labels = labels

        # Number of epochs
        self.epochs = epochs

        # Base, i.e., root, directory of logs relative to this project
        self.base = 'logs'

    def instance(self, model):
        """

        :param model:
        :return:
        """

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=3,
            mode='min',
            restore_best_weights=True
        )

        # validation_steps=math.floor(self.n_validating / self.batch_size)
        history = model.fit_generator(generator=self.training,
                                      steps_per_epoch=math.floor(self.n_training / self.batch_size),
                                      epochs=self.epochs,
                                      verbose=1,
                                      callbacks=[early_stopping],
                                      validation_data=self.validating)
        return history

    def run(self, hyperparameters):
        """

        :return:
        """

        # Metrics
        metrics = met.Metrics()

        # Architecture
        architecture = arc.Architecture()
        model: tf.keras.preprocessing.image.ImageDataGenerator = \
            architecture.core(hyperparameters=hyperparameters, metrics=metrics.classification(), labels=self.labels)

        history = Simple.instance(self, model=model)

        return history
