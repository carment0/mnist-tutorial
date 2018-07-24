from layers import Dense
import numpy as np

# Input dimension - 10
# Final output simension - 1
layer_dimens = [(10, 20), (20, 50), (50, 100), (100, 5), (5, 1)]
x = np.linspace(-1, 1, num=7 * 10).reshape(7, 10)

# Declare weights and biases for every layer
weights = dict()
bias = dict()
layers = dict()

for layer_num in range(len(layer_dimens)):
    # de-constructor dimension into in and out
    in_dim, out_dim = layer_dimens[layer_num]

    # randomly initialize some weights
    weights[layer_num] = np.random.randn(in_dim, out_dim)
    bias[layer_num] = np.zeros(out_dim)

    # create # of layers depending on # of dimensions
    layers[layer_num] = Dense()



# use forward in each layer to compute an output
for layer_num, dense in layers.iteritems():
    output = dense.forward(x, weights[layer_num], bias[layer_num])
    x = output

print x
