import math
import os

import tensorboard.plugins.hparams.api as hp
import tensorflow as tf

import src.data.reader as reader
import src.evaluating.metrics as met
import src.federal.federal as federal
import src.modelling.VGG19FE.architecture as arc
import src.modelling.VGG19FE.hyperparameters as hyp


class Complex:

    def __init__(self, training_, validating_, testing_, labels, epochs):
        """

        :param training_:
        :param validating_:
        :param testing_:
        :param labels:
        :param epochs:
        """

        # Variables
        variables = federal.Federal().variables()
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

        # Logs
        self.base = 'logs'


    def instance(self, hyperparameters_combination, combination, model):
        """

        :param hyperparameters_combination:
        :param combination:
        :param model:
        :return:
        """

        name = "C{}".format(str(combination).zfill(3))
        directory = os.path.join(self.base, name)

        model.fit_generator(generator=self.training,
                            steps_per_epoch=math.floor(self.n_training / self.batch_size),
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=directory, update_freq='epoch'),
                                       hp.KerasCallback(directory, hyp)],
                            validation_data=self.validating,
                            validation_steps=math.floor(self.n_validating / self.batch_size))


    def run(self):
        """

        :return:
        """

        # The hyperparameters priors & the list of hyperparameter combinations
        hyperparameters = hyp.Hyperparameters()
        alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization = hyperparameters.priors()
        hyperparameters_values = hyperparameters.values()

        # For TensorFlow's TensorBoard
        # noinspection PyUnresolvedReferences
        metrics = met.Metrics()
        with tf.summary.create_file_writer(logdir=self.base).as_default():
            hp.hparams_config(hparams=[alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization],
                              metrics=metrics.units())

        # Hyperparameters combinations enumerator
        combination = 0

        # For each combination of hyperparameters determine the Deep CNN Model
        architecture = arc.Architecture()
        for i in hyperparameters_values:
            model: tf.keras.preprocessing.image.ImageDataGenerator = \
                architecture.core(hyperparameters=i, metrics=metrics.classification(), labels=self.labels)

            Complex.instance(self, hyperparameters_combination=i, combination=combination, model=model)

            combination += 1
