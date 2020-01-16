import tensorflow as tf

import src.config as config


class Reader:

    def __init__(self, rescale=1. / 255):
        """
        :type rescale: float

        :param rescale: image integers scaling factor
        """
        self.rescale = rescale

        variables = config.Config().variables()
        self.rows = variables['images']['rows']
        self.columns = variables['images']['columns']
        self.batch_size = variables['modelling']['batch_size']


    @staticmethod
    def image_decoder(img):

        # Convert the compressed img to a 3D uint8 tensor
        img = tf.image.decode_png(contents=img, channels=3)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Hence
        return img


    @staticmethod
    def image_label_pairs(i, j):
        img = tf.io.read_file(i)
        img = Reader().image_decoder(img)
        return img, j


    def generator_tensorflow(self, data, labels):
        filenames = data['url'].values
        labelnames = data[labels].values

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labelnames))

        # 'cache/training/log'
        dataset = dataset.map(Reader().image_label_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


    def generator_keras(self, data, labels):
        base = tf.keras.preprocessing.image.ImageDataGenerator(rescale=self.rescale)
        return base.flow_from_dataframe(dataframe=data,
                                        directory=None,
                                        x_col='url',
                                        y_col=labels,
                                        target_size=(self.rows, self.columns),
                                        color_mode='rgb',
                                        class_mode='raw',
                                        batch_size=self.batch_size)
