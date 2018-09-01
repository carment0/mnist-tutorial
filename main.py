from implementation.layers import Dense
from implementation.layers import ReLU
import numpy as np

# # Input dimension - 10
# # Final output simension - 1
# layer_dimens = [(10, 20), (20, 50), (50, 100), (100, 5), (5, 1)]
# x = np.linspace(-1, 1, num=7 * 10).reshape(7, 10)

# # Declare weights and biases for every layer
# weights = dict()
# bias = dict()
# layers = dict()

# for layer_num in range(len(layer_dimens)):
#     # de-constructor dimension into in and out
#     in_dim, out_dim = layer_dimens[layer_num]

#     # randomly initialize some weights
#     weights[layer_num] = np.random.randn(in_dim, out_dim)
#     bias[layer_num] = np.zeros(out_dim)

#     # create # of layers depending on # of dimensions
#     layers[layer_num] = Dense()



# # use forward in each layer to compute an output
# for layer_num, dense in layers.iteritems():
#     output = dense.forward(x, weights[layer_num], bias[layer_num])
#     x = output

# print x

# _________
if __name__ == '__main__':
    N, D, H = 10, 8, 7
    x = np.random.rand(N, D)
    w = np.random.rand(D, H)
    b = np.random.rand(H,)

    l1 = Dense()
    l2 = ReLU()

    out = l2.forward(l1.forward(x, w, b))
    grad_out = np.random.rand(N, H)
    print grad_out.shape == out.shape

    grad_x, grad_w, grad_b = l1.backprop(l2.backprop(grad_out))
    print grad_x.shape == x.shape
    print grad_w.shape == w.shape
    print grad_b.shape == b.shape
