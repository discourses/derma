import tensorflow as tf

import config


class Architecture:

    def __init__(self):
        variables = config.Config().variables()
        self.image_dimension = variables['images']['image_dimension']

    def baseline(self):
        """
        Sets-up the baseline architecture

        :return:
        """

        # Base Model
        base = tf.keras.applications.VGG19(include_top=False, input_shape=self.image_dimension, weights='imagenet')
        base.trainable = False

        return base

    @staticmethod
    def layers(hyperparameters, labels, metrics=None):

        # Base
        base = Architecture().baseline()

        # Flattening Object
        flatten = tf.keras.layers.Flatten()

        # The fully connected layers
        alpha = tf.keras.layers.Dense(hyperparameters['alpha_units'], activation='relu', name='Alpha')
        alpha_drop = tf.keras.layers.Dropout(rate=hyperparameters['alpha_drop_rate'], name='AlphaDrop')
        beta = tf.keras.layers.Dense(hyperparameters['beta_units'], activation='relu', name='Beta')
        beta_drop = tf.keras.layers.Dropout(hyperparameters['beta_drop_rate'], name='BetaDrop')

        # The classification layer
        classifier = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)

        # Build
        model = tf.keras.models.Sequential([base, flatten, alpha, alpha_drop,
                                            beta, beta_drop, classifier])

        # Labels. Case
        # one-hot-code: categorical_crossentropy
        # integers: sparse_categorical_crossentropy
        if metrics is None:
            model.compile(optimizer=hyperparameters['optimization'],
                          loss=tf.keras.losses.categorical_crossentropy)
        else:
            model.compile(optimizer=hyperparameters['optimization'],
                          metrics=metrics,
                          loss=tf.keras.losses.categorical_crossentropy)

        print(base.summary())
        print(model.summary())

        return model
