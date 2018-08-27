from softmax import Softmax
import numpy as np
import unittest
from gradient_check import eval_numerical_gradient
from implementation.models import avg_loss

class SoftmaxTest(unittest.TestCase):
    def setUp(self):
        self.layer = Softmax()

    def test_backprop(self):
        dummy_scores = np.random.rand(5,5)
        dummy_labels = np.array([0, 1, 2, 3, 4])

        expected_grad = eval_numerical_gradient(
            lambda x: avg_loss(self.layer.forward(x), dummy_labels),
            dummy_scores)

        grad = self.layer.backprop(dummy_labels)

        np.testing.assert_almost_equal(expected_grad, grad, decimal=5)

    def tearDown(self):
        pass
