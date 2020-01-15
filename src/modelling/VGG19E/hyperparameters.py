import tensorboard.plugins.hparams.api as hp


class Hyperparameters:

    def __init__(self):

        self.name = "Hyperparameters"

    @staticmethod
    def priors():
        alpha_units = hp.HParam('num_units', hp.Discrete([512]))
        alpha_drop_rate = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))

        beta_units = hp.HParam('num_units', hp.Discrete([512]))
        beta_drop_rate = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))

        optimization = hp.HParam('optimizer', hp.Discrete(['adam']))

        return alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization

    @staticmethod
    def values():
        alpha_units, alpha_drop_rate, beta_units, beta_drop_rate, optimization = Hyperparameters.priors()

        combinations = [{'alpha_drop_rate': i,
                         'alpha_units': j,
                         'beta_drop_rate': x,
                         'beta_units': y,
                         'optimization': opt}

                        for i in [alpha_drop_rate.domain.sample_uniform() for _ in range(2)]
                        for j in alpha_units.domain.values
                        for x in [beta_drop_rate.domain.sample_uniform() for _ in range(2)]
                        for y in beta_units.domain.values
                        for opt in optimization.domain.values]

        return combinations
