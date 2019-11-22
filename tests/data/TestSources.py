import src.data.Sources as Sources


class TestSources:

    def test_truth(self):
        _, n_truth = Sources.Sources.truth()
        assert n_truth != 0, "The ground truth labels file should not be empty"

    def test_metadata(self):
        _, n_metadata = Sources.Sources.metadata()
        assert n_metadata != 0, "The metadata file should not be empty"
