import src.data.source as source


class TestSource:

    def test_inventory(self):
        inventory = source.Source().inventory()
        assert inventory.shape[0] != 0, "The images inventory file must not be empty"

    def test_url(self):
        inventory = source.Source().inventory()
        inventory = source.Source().url(inventory)

        assert inventory.shape[0] != 0, "The data frame must not be empty"
        assert any(inventory.columns.values == 'url'), "The url method/function should add a url field to inventory"
