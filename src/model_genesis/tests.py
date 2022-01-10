import unittest

import numpy as np

from src.model_genesis.tf_data_hdf5 import pick_random_origin
from src.model_genesis.utils import degrade_image


class TestModelGenesis(unittest.TestCase):
    def test_pick_random_origins(self):
        mask = np.zeros((128, 128, 128))
        mask[108:, 108:, 127] = 1
        origin = pick_random_origin((64, 64, 64), mask)
        mask_cropped = mask[origin[0]:origin[0] + 64, origin[1]:origin[1] + 64,
                            origin[2]:origin[2] + 64]
        assert mask_cropped.shape == (64, 64, 64)

        mask = np.zeros((128, 128, 128))
        mask[10:, 10:, 0] = 1
        origin = pick_random_origin((64, 64, 64), mask)
        mask_cropped = mask[origin[0]:origin[0] + 64, origin[1]:origin[1] + 64,
                            origin[2]:origin[2] + 64]
        assert mask_cropped.shape == (64, 64, 64)

    def test_degrad_image(self):
        image = np.random.uniform(size=(64, 64, 64, 2))
        degraded_image = degrade_image(image)
        assert degraded_image.shape == image.shape


if __name__ == "__main__":
    unittest.main()