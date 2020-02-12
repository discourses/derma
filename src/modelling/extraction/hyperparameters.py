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

    def __init__(self, hyperparameters_dict: dict):
        """
        Herein the constructor initialises global variables
        """
        # variables: dict = config.Config().variables()

        # model_extraction: dict = variables['model']['extraction']
        # self.num_units: list = model_extraction['num_units']
        # self.dropout: list = model_extraction['dropout']
        # self.opt: list = model_extraction['opt']

        self.alpha_units = hyperparameters_dict['alpha']['units']
        self.alpha_dropout = hyperparameters_dict['alpha']['dropout']
        self.beta_units = hyperparameters_dict['beta']['units']
        self.beta_dropout = hyperparameters_dict['beta']['dropout']
        self.opt: list = hyperparameters_dict['opt']

    def priors(self) -> (hp.HParam, hp.HParam, hp.HParam, hp.HParam, hp.HParam):
        """
        Initialises the set of values per hyperparameter type
        :return:
        """
        alpha_units = hp.HParam('num_units', hp.Discrete(self.alpha_units))
        alpha_dropout = hp.HParam('dropout', hp.Discrete(self.alpha_dropout))

        beta_units = hp.HParam('num_units', hp.Discrete(self.beta_units))
        beta_dropout = hp.HParam('dropout', hp.Discrete(self.beta_dropout))

        optimization = hp.HParam('optimizer', hp.Discrete(self.opt))

        return alpha_units, alpha_dropout, beta_units, beta_dropout, optimization

    def values(self) -> list:
        """
        Creates unique combinations of hyperparameters
        :return:
            combinations: A list of dictionaries, wherein each dictionary is a unique combination
            of hyperparameters.  Each combination estimates a distinct/single model.
        """
        alpha_units, alpha_dropout, beta_units, beta_dropout, optimization = self.priors()

        combinations = [{'alpha_dropout': i,
                         'alpha_units': j,
                         'beta_dropout': x,
                         'beta_units': y,
                         'optimization': opt}

                        for i in alpha_dropout.domain.values
                        for j in alpha_units.domain.values
                        for x in beta_dropout.domain.values
                        for y in beta_units.domain.values
                        for opt in optimization.domain.values]

        return combinations
