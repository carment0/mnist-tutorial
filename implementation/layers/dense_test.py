from dense import Dense
import numpy as np
import unittest

class DenseTest(unittest.TestCase):
    # When a setUp() method is defined, the test runner will run that method
    # prior to each test. Likewise, if a tearDown() method is defined, the
    # test runner will invoke that method after each test.
    def setUp(self):
        self.N = 2 # number of inputs
        self.D = 3 # input dimension
        self.H = 4 # output/hidden dimension
        self.layer = Dense() # defines the layer we are testing

    def tearDown(self):
        pass

    def test_one_plus_one(self):
        self.assertEqual(1 + 1, 2)

    def test_one_plus_two(self):
        self.assertEqual(1 + 2, 3)

    def test_forward(self):
        # `numpy.linspace(start, stop, num=50...)` returns an array of evenly
        # spaced numbers over a specified interval.
        # `num=` Number of samples to generate. Default is 50. Must be non-negative.
        x = np.linspace(-1, 1, num=self.N * self.D).reshape(self.N, self.D)
        w = np.linspace(-0.5, 0.5, num=self.D * self.H).reshape(self.D, self.H)
        b = np.linspace(-0.5, 0.5, num=self.H)

        output = self.layer.forward(x, w, b)
        expected_output = np.dot(x, w) + b

        np.testing.assert_array_almost_equal(expected_output, output, decimal=9)

# TODO: line 28-30 why those ranges?