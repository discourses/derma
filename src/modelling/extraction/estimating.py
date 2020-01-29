import math
import os

import tensorflow as tf

import config
import src.data.pipelines as pipelines
import src.evaluation.measures as measures
import src.modelling.extraction.architecture as arc


class Estimating:

    def __init__(self):
        """

        """

        # Variables
        variables = config.Config().variables()
        self.batch_size = variables['modelling']['batch_size']

    def parameters(self, model, labels, epochs, training_, validating_, network_checkpoints_path):
        """
        validation_steps=math.floor(self.validating_.shape[0] / self.batch_size)

        :param model:
        :param labels:
        :param epochs:
        :param training_:
        :param validating_:
        :param network_checkpoints_path:
        :return:
        """

        # Data
        pipeline = pipelines.Pipelines()
        training = pipeline.generator_tensorflow(training_, labels)
        validating = pipeline.generator_tensorflow(validating_, labels)

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=3,
            mode='min',
            restore_best_weights=True
        )

        # Checkpoints
        model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(network_checkpoints_path, 'model_{epoch}.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch')

        # History
        history = model.fit_generator(generator=training,
                                      steps_per_epoch=math.floor(training_.shape[0] / self.batch_size),
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=[early_stopping, model_checkpoints],
                                      validation_data=validating)

        return history

    def network(self, hyperparameters, labels, epochs, training_, validating_, testing_, network_checkpoints_path):
        """

        :return:
        """

        # Architecture
        architecture = arc.Architecture()
        model: tf.keras.preprocessing.image.ImageDataGenerator = \
            architecture.layers(hyperparameters=hyperparameters, labels=labels)

        # Estimate the parameters of the network described by model's architecture.  The parameters
        # function saves the models at the end of every epoch.  And, it returns the loss history
        history = self.parameters(model, labels, epochs,
                                  training_, validating_, network_checkpoints_path)

        measures.Measures().calculate(history=history,
                                      network_checkpoints_path=network_checkpoints_path,
                                      training_=training_,
                                      validating_=validating_,
                                      testing_=testing_,
                                      labels=labels)
