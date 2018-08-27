import numpy as np
from loss import entropy_loss, avg_loss

class Sequential(object):
    def __init__(self):
        self.layers = []
        self.act = []
        self.weight = []
        self.bia = []

    def add(self, layer, act, weight_dim):
        self.layers.append(layer)
        self.act.append(act)

        D, H = weight_dim
        self.weight.append(np.random.randn(D, H))
        self.bia.append(np.ones((H,)))

    def predict(self, input):
        x = input
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x, self.weight[i], self.bia[i])
            # print "dense"
            # print x
            x = self.act[i].forward(x)
            # print "act"
            # print x
        return x

    def calculate_loss(self, input, expected):
        predicted = self.predict(input)
        return avg_loss(predicted, expected)
