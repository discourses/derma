import src.data.Sources as Sources
import src.data.DermoscopicImages as DermoscopicImages


class TestDermoscopicImages:

    def test_states(self):
        listing, _, _ = Sources.Sources().summary()
        states = DermoscopicImages.DermoscopicImages().states(listing['image'])
        assert states.status.all()
