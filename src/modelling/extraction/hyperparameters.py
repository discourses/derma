"""Module hyperparameters"""
import tensorboard.plugins.hparams.api as hp

import config


class Hyperparameters:
    """
    Class Hyperparameters

    The numerical values of this program must be converted to YAML dictionary values, and subsequently read-in

    For the case wherein
        alpha_drop_rate = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
    the combinations code would be
        for i in [alpha_drop_rate.domain.sample_uniform() for _ in range(n)]
    whereby n is the number of values that should be sampled from the range.
    """

    def __init__(self):

        variables = config.Config().variables()
        model_extraction = variables['model']['extraction']
        self.num_units = model_extraction['num_units']
        self.dropout = model_extraction['dropout']
        self.opt = model_extraction['opt']
        self.name = "Hyperparameters"


    def priors(self):
        alpha_units = hp.HParam('num_units', hp.Discrete(self.num_units))
        alpha_drop_rate = hp.HParam('dropout', hp.Discrete(self.dropout))

        beta_units = hp.HParam('num_units', hp.Discrete(self.num_units))
        beta_drop_rate = hp.HParam('dropout', hp.Discrete(self.dropout))

        optimization = hp.HParam('optimizer', hp.Discrete(self.opt))

        return alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization


    def values(self):
        alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization = self.priors()

        combinations = [{'alpha_drop_rate': i,
                         'alpha_units': j,
                         'beta_drop_rate': x,
                         'beta_units': y,
                         'optimization': opt}

                        for i in alpha_drop_rate.domain.values
                        for j in alpha_units.domain.values
                        for x in beta_drop_rate.domain.values
                        for y in beta_units.domain.values
                        for opt in optimization.domain.values]

        return combinations
