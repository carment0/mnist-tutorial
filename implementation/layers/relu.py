import numpy as np
from gradient_check import eval_numerical_gradient_for_matrix

class ReLU(object):
    def __init__(self):
        self.input = None


    def forward(self, x):
        self.input = x
        return np.maximum(0, self.input)


    def backprop(self, grad_f):
        x_lambda = lambda x: self.forward(x)
        grad_x = eval_numerical_gradient_for_matrix(x_lambda, self.input, grad_f)

        return grad_x

# if __name__ == '__main__':
#     dummy = ReLU()
#     x = [[1, -4, 6, -3], [3, -5, -8 , 5]]
#     arr = np.array(x)
#     print dummy.forward(arr)

if __name__ == '__main__':
    """
    Num of Sample: N
    Hidden Dimension: H
    """
    N, H = 10, 7
    x = np.random.rand(N, H)

    layer = ReLU()
    output = layer.forward(x)
    print output
    print output.shape == (N, H)

    grad_out = np.random.rand(N, H)
    grad_x = layer.backprop(grad_out)

    print grad_x.shape

