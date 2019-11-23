import os

import src.data.Sources as Sources
import src.data.images as images


def test_states():
    print("here " + os.getcwd())
    dataset, _, _ = Sources.Sources().summary()
    states = images.states(dataset[['image']])
    all_random_images_exist = states[['status']].apply(lambda j: j == 200).all(axis=0, bool_only='bool')[0]
    assert all_random_images_exist
