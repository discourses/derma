import tensorboard.plugins.hparams.api as hp
import tensorflow as tf

import src.cfg.cfg as cfg


class Architecture:

    def __init__(self):
        self.name = 'Extraction'
        variables = cfg.Cfg().variables()
        self.image_dimension = variables['images']['image_dimension']

    @staticmethod
    def hyperparameters():

        alpha_units = hp.HParam('num_units', hp.Discrete([512]))
        alpha_drop_rate = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))

        beta_units = hp.HParam('num_units', hp.Discrete([512]))
        beta_drop_rate = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))

        optimization = hp.HParam('optimizer', hp.Discrete(['adam']))

        return alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization

    def core(self, hyp, metrics, labels):

        # Base Model
        base = tf.keras.applications.VGG19(include_top=False, input_shape=self.image_dimension, weights='imagenet')
        base.trainable = False

        # Flattening Object
        flatten = tf.keras.layers.Flatten()

        # The fully connected layers
        alpha = tf.keras.layers.Dense(hyp['alpha_units'], activation='relu', name='Alpha')
        alpha_drop = tf.keras.layers.Dropout(rate=hyp['alpha_drop_rate'], name='AlphaDrop')
        beta = tf.keras.layers.Dense(hyp['beta_units'], activation='relu', name='Beta')
        beta_drop = tf.keras.layers.Dropout(hyp['beta_drop_rate'], name='BetaDrop')

        # The classification layer
        classifier = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)

        # Build
        model = tf.keras.models.Sequential([base, flatten, alpha, alpha_drop,
                                            beta, beta_drop, classifier])

        # Labels. Case
        # one-hot-code: categorical_crossentropy
        # integers: sparse_categorical_crossentropy
        model.compile(optimizer=hyp['optimization'],
                      metrics=metrics,
                      loss=tf.keras.losses.categorical_crossentropy)

        print(base.summary())
        print(model.summary())

        return model

    @staticmethod
    def hyper():

        alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization = Architecture.hyperparameters()

        hyper = [{'alpha_drop_rate': i,
                  'alpha_units': j,
                  'beta_drop_rate': x,
                  'beta_units': y,
                  'optimization': opt}

                 for i in [alpha_drop_rate.domain.sample_uniform() for _ in range(2)]
                 for j in alpha_units.domain.values
                 for x in [beta_drop_rate.domain.sample_uniform() for _ in range(2)]
                 for y in beta_units.domain.values
                 for opt in optimization.domain.values]

        return hyper
