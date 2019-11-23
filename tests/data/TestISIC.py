import src.data.ISIC as ISIC


class TestISIC:

    def test_api(self):
        assert len(ISIC.ISIC().api()) != 0
