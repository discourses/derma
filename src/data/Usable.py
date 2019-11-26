import src.data.Sources as Sources


class Usable:
    def __init__(self):
        self.name = 'Feasible'

    @staticmethod
    def records():
        # The metadata & ground truth (listing), the names of the label columns, and
        # the names of the metadata fields
        listing, labels, fields = Sources.Sources().summary()

        # The minimum number of instances, i.e., data points, required
        # per class.  Presently based on the minimum number of instances per class
        # required for the stratification of imbalanced data sets w.r.t. SciKit Learn, etc..
        min_instances_per_class = 2

        # In terms of the data in question, how many instances are there per class?
        instances_per_class = listing[labels].sum()

        # Hence, are there any outlying classes?
        outliers = instances_per_class[instances_per_class < min_instances_per_class]
        inadmissible = listing[listing[outliers.index.values].any(axis=1)].index
        [labels.remove(i) for i in outliers.index.values]

        # If yes, the associated rows & columns are dropped.
        return listing.drop(inadmissible, axis=0).drop(outliers.index.values, axis=1), \
            labels, fields
