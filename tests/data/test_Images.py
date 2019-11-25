import src.data.Sources as Sources
import src.data.Images as Images


class TestImages:

    def test_states(self):
        listing, _, _ = Sources.Sources().summary()
        states = Images.Images().states(listing['image'])
        assert states.status.all(), "It seems one or more images do not exists in the repository.  " \
                                    "The missing images are listed in the logs."
