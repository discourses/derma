import src.data.Archive as Archive


class TestArchive:

    def test_images(self):
        assert len(Archive.Archive().images()) != 0, "There are no images in the ISIC Archive"
