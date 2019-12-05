import src.data.usable as usable


class TestUsable:

    def test_summary(self):
        listing, labels, fields = usable.Usable().summary()

        assert len(labels) != 0, "At least one label column must exist"
        assert len(fields) != 0, "Missing field names"
        assert listing.image.unique().shape[0] == listing.shape[0], "Each image name can occur once only"
        assert listing[labels].sum(axis=1).all(), "Each image must be associated with a single class only"
