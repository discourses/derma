import src.data.sources as sources


class TestSources:

    def test_truth(self):
        truth, n_truth = sources.Sources().truth()
        assert truth.shape[0] != 0, "The ground truth labels file should not be empty"
        assert n_truth != 0, "The ground truth labels file should have more than zero records"
        assert truth.shape[0] == n_truth, "The variable 'n_truth' is the number of records in data frame 'truth', " \
                                          " therefore truth.shape[0] & n_truth must be equal."

    def test_metadata(self):
        metadata, n_metadata = sources.Sources().metadata()
        assert metadata.shape[0] != 0, "The metadata file should not be empty"
        assert n_metadata != 0, "The metadata file should have more than zero records"
        assert metadata.shape[0] == n_metadata, "The variable 'n_metadata' is the number of records in " \
                                                "data frame 'metadata', therefore metadata.shape[0] & " \
                                                "n_metadata must be equal."

    def test_summary(self):
        _, n_truth = sources.Sources().truth()
        _, n_metadata = sources.Sources().metadata()
        listing, labels, fields = sources.Sources().summary()

        assert n_truth == listing.shape[0], "The number of ground truth & listing data points must be equal"
        assert n_metadata == listing.shape[0], "The number of medata & listing data points must be equal"
        assert len(labels) != 0, "At least one label column must exist"
        assert len(fields) != 0, "Missing field names"
        assert listing.image.unique().shape[0] == listing.shape[0], "Each image name can occur once only"
        assert listing[labels].sum(axis=1).all(), "Each image must be associated with a single class only"
