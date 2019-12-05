import src.data.archive as archive


class TestArchive:

    def test_images(self):
        assert len(archive.Archive().images()) != 0, "There are no images in the ISIC Archive"
