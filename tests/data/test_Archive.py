import src.data.Archive as Archive


class TestArchive:

    def test_imges(self):
        assert len(Archive.Archive().images()) != 0
