import os

import src.data.inventory as inventory
import src.data.images as images


def test_states():
    print("here " + os.getcwd())
    dataset, _, _ = inventory.inventory()
    states = images.states(dataset[['image']])
    all_random_images_exist = states[['status']].apply(lambda j: j == 200).all(axis=0, bool_only='bool')[0]
    assert all_random_images_exist
