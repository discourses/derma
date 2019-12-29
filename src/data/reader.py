import tensorflow as tf

import src.cfg.cfg as cfg


class Reader:

    def __init__(self, rescale=1./255):
        """
        :type rescale: float

        :param rescale: image integers scaling factor
        """
        self.rescale = rescale

        variables = cfg.Cfg().variables()
        self.rows = variables['images']['rows']
        self.columns = variables['images']['columns']
        self.batch_size = variables['modelling']['batch_size']

    def generator(self, data, labels):
        base = tf.keras.preprocessing.image.ImageDataGenerator(rescale=self.rescale)
        return base.flow_from_dataframe(dataframe=data,
                                        directory='/content/img/',
                                        x_col='url',
                                        y_col=labels,
                                        target_size=(self.rows, self.columns),
                                        color_mode='rgb',
                                        class_mode='raw',
                                        batch_size=self.batch_size)
