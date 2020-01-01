import matplotlib.pyplot as plt


class Graphing:

    def __init__(self):
        self.name = 'Graphing'

        self.colours = plt.rcParams['axes.prop_cycle'].by_key()['color']


    def plot_metrics(self, history):

        metrics = ['loss', 'auc', 'precision', 'recall']

        for n, metric in enumerate(metrics):

            name = metric.replace("_", " ").capitalize()

            plt.subplot(2, 2, n + 1)

            plt.plot(history.epoch,
                     history.history[metric], color=self.colours[0], label='Train')

            plt.plot(history.epoch,
                     history.history['val_' + metric], color=self.colours[0], linestyle="--", label='Val')

            plt.xlabel('Epoch')
            plt.ylabel(name)

            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
