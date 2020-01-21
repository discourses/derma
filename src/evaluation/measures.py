import src.evaluation.confusion as confusion

import src.data.pipelines as pipelines


class Predictions:

    def __init__(self):

        self.name = 'Crazy'


    @staticmethod
    def calculate(model, data, labels, name, path):

        # name: training, validating, testing
        print(name)

        # Truth
        truth = data[labels].values

        # Predictions
        pipeline = pipelines.Pipelines()
        dataset = pipeline.generator_tensorflow(data, labels)
        plausibilities = model.predict(dataset)

        confusion.Confusion().calculate(plausibilities, truth, labels, path)
