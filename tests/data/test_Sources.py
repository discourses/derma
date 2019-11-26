import src.data.Sources as Sources


class TestSources:

    def test_truth(self):
        _, n_truth = Sources.Sources().truth()
        assert n_truth != 0, "The ground truth labels file should not be empty"

    def test_metadata(self):
        _, n_metadata = Sources.Sources().metadata()
        assert n_metadata != 0, "The metadata file should not be empty"

    def test_summary(self):
        _, n_truth = Sources.Sources().truth()
        _, n_metadata = Sources.Sources().metadata()
        listing, labels, fields = Sources.Sources().summary()

        assert n_truth == listing.shape[0], "The number of ground truth & listing data points must be equal"
        assert n_metadata == listing.shape[0], "The number of medata & listing data points must be equal"
        assert len(labels) != 0, "At least one label column must exist"
        assert len(fields) != 0, "Missing field names"
        assert listing[labels].sum(axis=1).all(), "Each image must be associated with a single class only"
