import math

import pandas as pd

import config
import src.data.pipelines as pipelines
import src.evaluation.measures as measures
import src.evaluation.monitors as monitors


class Estimating:

    def __init__(self):
        """

        """

        # Variables
        variables = config.Config().variables()
        self.batch_size = variables['modelling']['batch_size']
        self.patience = variables['modelling']['early_stopping_patience']

    def parameters(self, model, labels: list, epochs: int, training_: pd.DataFrame, validating_: pd.DataFrame,
                   network_checkpoints_path: str):
        """

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
        early_stopping = monitors.Monitors().early_stopping()

        # Checkpoints
        model_checkpoints = monitors.Monitors().model_checkpoints(network_checkpoints_path=network_checkpoints_path)

        # Steps per epoch [Re-design: cf. validation_steps, and steps in measures.Measures().prediction]
        steps_per_epoch = math.ceil(training_.shape[0] / self.batch_size)

        # Validation steps [Re-design: cf. steps_per_epoch, and steps in measures.Measures().prediction]
        validation_steps = math.ceil(validating_.shape[0] / self.batch_size)

        # History
        print('For: ' + network_checkpoints_path)
        history = model.fit_generator(generator=training, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                      verbose=1, callbacks=[early_stopping, model_checkpoints],
                                      validation_data=validating, validation_steps=validation_steps,
                                      validation_freq=1)

        return history

    def network(self, model, labels: list, epochs: int,
                training_: pd.DataFrame, validating_: pd.DataFrame, testing_: pd.DataFrame,
                network_checkpoints_path: str):
        """

        :param model:
        :param labels:
        :param epochs:
        :param training_:
        :param validating_:
        :param testing_:
        :param network_checkpoints_path:
        :return:
        """

        # Estimate the parameters of the network described by model's architecture.  The parameters
        # function saves the models at the end of every epoch.  And, it returns the loss history

        history = self.parameters(model, labels, epochs,
                                  training_, validating_, network_checkpoints_path)

        measures.Measures().calculate(history=history, network_checkpoints_path=network_checkpoints_path,
                                      training_=training_, validating_=validating_, testing_=testing_,
                                      labels=labels)
