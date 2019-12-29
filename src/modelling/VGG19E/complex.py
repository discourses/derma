import math
import os

import tensorboard.plugins.hparams.api as hp
import tensorflow as tf

import src.cfg.cfg as cfg
import src.data.reader as reader
import src.modelling.VGG19E.architecture as arc
import src.modelling.metrics as met


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


    def instance(self, hyp, combination, model):
        """

        :param hyp:
        :param combination:
        :param model:
        :return:
        """

        name = "C{}".format(str(combination).zfill(3))
        directory = os.path.join(self.base, name)

        model.fit_generator(generator=self.training,
                            steps_per_epoch=math.floor(len(self.training) / self.batch_size),
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=directory, update_freq='epoch'),
                                       hp.KerasCallback(directory, hyp)],
                            validation_data=self.validating,
                            validation_steps=math.floor(len(self.validating) / self.batch_size))


    def run(self):
        """

        :return:
        """

        # Metrics
        metrics = met.Metrics()

        # Architecture
        architecture = arc.Architecture()
        hyper = architecture.hyper()

        # The hyperparameters
        alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization = architecture.hyperparameters()

        # For TensorFlow's TensorBoard
        # noinspection PyUnresolvedReferences
        with tf.summary.create_file_writer(logdir=self.base).as_default():
            hp.hparams_config(hparams=[alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization],
                              metrics=metrics.units())

        # Hyperparameters combinations enumerator
        combination = 0

        # For each combination of hyperparameters determine the Deep CNN Model
        for hyp in hyper:
            model: tf.keras.preprocessing.image.ImageDataGenerator = \
                architecture.core(hyp=hyp, metrics=metrics.classification(), labels=self.labels)

            Complex.instance(self, hyp=hyp, combination=combination, model=model)

            combination += 1
