import os

import src.config as config
import src.modelling.extraction.estimating as estimating
import src.modelling.extraction.hyperparameters as hyperparameters


class Steps:

    def __init__(self):

        # Variables
        variables = config.Config().variables()
        self.model_checkpoints_path = variables['modelling']['model_checkpoints_path']


    @staticmethod
    def partitions(path):
        """

        :param path:
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)


    def proceed(self, labels, epochs, training_, validating_, testing_):

        hyp = hyperparameters.Hyperparameters()
        est = estimating.Estimating()

        self.partitions(self.model_checkpoints_path)

        for i in range(len(hyp.values())):
            # Ensure that the model checkpoints for a combination of hyperparameters are saved
            # to a distinct directory that EXISTS
            network_checkpoints_path = os.path.join(self.model_checkpoints_path, str(i).zfill(4))
            self.partitions(network_checkpoints_path)

            # History of losses
            est.network(hyperparameters=hyp.values()[i], labels=labels, epochs=epochs,
                        training_=training_, validating_=validating_, testing_=testing_,
                        network_checkpoints_path=network_checkpoints_path)
