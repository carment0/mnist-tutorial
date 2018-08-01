import numpy as np

class Dense(object):
    def __init__(self):
        self.input = None
        self.weight = None
        self.bia = None

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


if __name__ == '__main__':
    dummy = Dense()
    # input is N by D, where D is the input dimension.
    # (In this example N=2, D=3)
    # if you have multiple "examples" you can stack them into one matrix and feed it in
    x = np.array([[1, 2],
                  [1, 2],
                  [1, 2]]) # Shape: (3, 2)

    # weight is D by H, where H is the hidden unit dimension
    w = np.array([[1, 2, 3],
                  [4, 5, 6]]) # Shape (2, 3)

    # bias is always 1 by D, where D is the output dimension
    b = np.array([[1, 1, 1]]) # Shape: (1, 3)

    # What is the shape of the output? (1, 3)!
    print dummy.forward(x, w, b)

# TODO: line 40 does it have to have the same input to output
