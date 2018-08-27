import numpy as np
from layers import Dense, ReLU, Softmax
from models import Sequential


if __name__ == '__main__':
    seq = Sequential()
    seq.add(Dense(), ReLU(), [5, 10])
    seq.add(Dense(), ReLU(), [10, 15])
    seq.add(Dense(), ReLU(), [15, 30])
    seq.add(Dense(), Softmax(), [30, 2])
    input = np.ones((5, 5))
    expected = np.array([1, 1, 1, 1, 1])
    print seq.calculate_loss(input, expected)
    # => (5, 2) => (N, O)
    # prediction is the output (O) of the modal
    # O is the number of category

# how do we compare:
# output: [1, 2, 3, 4] to expected: 0? we can't use the same square (MAE) loss function
# instead we are going to use cross (categorical) entropy loss
# loss = -log(prob(y))

# TODO: loss

# Back propagation




