import tensorflow as tf

import src.federal.federal as federal


class Architecture:

    def __init__(self):
        self.name = 'Extraction'
        variables = federal.Federal().variables()
        self.image_dimension = variables['images']['image_dimension']


    def core(self, hyperparameters, metrics, labels):
        # Base Model
        base = tf.keras.applications.VGG19(include_top=False, input_shape=self.image_dimension, weights='imagenet')
        base.trainable = False

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
        model.compile(optimizer=hyperparameters['optimization'],
                      metrics=metrics,
                      loss=tf.keras.losses.categorical_crossentropy)

        print(base.summary())
        print(model.summary())

        return model
