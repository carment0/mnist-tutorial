from relu import ReLU
import numpy as np
import unittest


class ReluTest(unittest.TestCase):
    def setUp(self):
        self.layer = ReLU()

    def tearDown(self):
        pass

    def test_forward(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        output = self.layer.forward(x)
        x[x < 0] = 0
        expected_output = x

        np.testing.assert_array_almost_equal(expected_output, output, decimal=9)
