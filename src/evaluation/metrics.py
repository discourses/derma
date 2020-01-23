import tensorboard.plugins.hparams.api as hp
import tensorflow as tf


class Metrics:

    def __init__(self):
        self.name = 'Metrics'

    @staticmethod
    def definitions_keras():
        metrics = [tf.keras.metrics.TruePositives(name='tp'),
                   tf.keras.metrics.FalsePositives(name='fp'),
                   tf.keras.metrics.TrueNegatives(name='tn'),
                   tf.keras.metrics.FalseNegatives(name='fn'),
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'),
                   tf.keras.metrics.AUC(name='auc')]

        return metrics

    @staticmethod
    def definitions_tensorflow():
        metrics = [hp.Metric('tp'),
                   hp.Metric('fp'),
                   hp.Metric('tn'),
                   hp.Metric('fn'),
                   hp.Metric('precision'),
                   hp.Metric('recall'),
                   hp.Metric('auc')]

        return metrics
