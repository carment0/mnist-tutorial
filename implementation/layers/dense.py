import numpy as np
from gradient_check import eval_numerical_gradient_for_matrix

class Dense(object):
    def __init__(self):
        self.input = None
        self.weight = None
        self.bias = None


    def forward(self, x, w, b):
        """ Perform forward propagation in the Dense Layer

        Args:
            x (np.ndarray): Input data in matrix form.
            w (np.ndarray): Weight matrix for the layer.
            b (np.ndarray): Bias array.

        Returns:
            output (p.ndarray)
        """
        self.input = x
        self.weight = w
        self.bias = b
        return np.dot(x, w) + b


    def backprop(self, grad_f):
        x_lambda = lambda x: self.forward(x, self.weight, self.bias)
        grad_x = eval_numerical_gradient_for_matrix(x_lambda, self.input, grad_f)

        b_lambda = lambda b: self.forward(self.input, self.weight, b)
        grad_b = eval_numerical_gradient_for_matrix(b_lambda, self.bias, grad_f)

        w_lambda = lambda w: self.forward(self.input, w, self.bias)
        grad_w = eval_numerical_gradient_for_matrix(w_lambda, self.weight, grad_f)

        return grad_x, grad_w, grad_b

# if __name__ == '__main__':
#     dummy = Dense()
#     # input is N by D, where D is the input dimension.
#     # (In this example N=2, D=3)
#     # if you have multiple "examples" you can stack them into one matrix and feed it in
#     x = np.array([[1, 2],
#                   [1, 2],
#                   [1, 2]]) # Shape: (3, 2)

#     # weight is D by H, where H is the hidden unit dimension
#     w = np.array([[1, 2, 3],
#                   [4, 5, 6]]) # Shape (2, 3)

#     # bias is always 1 by D, where D is the output dimension
#     b = np.array([[1, 1, 1]]) # Shape: (1, 3)

#     # What is the shape of the output? (1, 3)!
#     print dummy.forward(x, w, b)

# # TODO: line 40 does it have to have the same input to output

if __name__ == '__main__':
    """
    Num of Sample: N
    Input Dimension: D
    Hidden Dimension: H

    Shape of x: (N, D)
    Shape of w: (D, H)
    Shape of b: (H, )
    """
    N, D, H = 10, 8, 7
    x = np.random.rand(N, D)
    w = np.random.rand(D, H)
    b = np.random.rand(H,)

    layer = Dense()
    output = layer.forward(x, w, b)
    print output
    print output.shape == (N, H)

    grad_out = np.random.rand(N, H)
    grad_x, grad_w, grad_b = layer.backprop(grad_out)

    print grad_x.shape
    print grad_w.shape
    print grad_b.shape