import numpy as np
from loss import entropy_loss, avg_loss

class Sequential(object):
    def __init__(self):
        self.layers = {}
        self.activations = {}
        self.weights = {}
        self.biases = {}

    def add(self, layer, act, layer_dim, weight_scale=1e-2):
        """
        Args:
            layer (Dense): Just dense layer for now
            act (Relu|Softmax): Activation layer
            layer_dim (Tuple): A tuple of two integers, indicating the dimensions
        """

        idx = len(self.layers) # layer index: 0, 1, 2..
        self.weights[idx] = np.random.normal(loc=0, scale=weight_scale, size=layer_dim)
        self.biases[idx] = np.zeros(layer_dim[1],)
        self.layers[idx] = layer
        self.activations[idx] = act

    def compile(self, optimizer=None, loss_func=None):
        """
        Args:
            optimizer (GradientDescent): An optimizer object for training the model
            loss func (Function): A funcation for computing loss
        """
        self.optimizer = optimizer
        self.loss_func = loss_func

    def loss(self, x, y):
        """
        Args:
            x (numpy.ndarray): Input to the neural network, aka training data
            y (numpy.ndarray): Expected output of the neural network, aka training data
        """
        if self.loss_func is None:
            raise Exception("Please provide a loss function using compile() before calling loss")

        for i in range(len(self.layers)):
            x = self.layers[i].forward(x, self.weight[i], self.biases[i])
            x = self.activations[i].forward(x)

        # redult from the last softmax activation layer
        predictied_probs = x
        loss = self.loss_func(predictied_probs, y)

        # now we are ready for back prop
        grad_out = y
        grad_weight = {}
        grad_biases = {}
        for i in reversed(range(len(self.layers))):
            grad_out = self.activations[i].backprop(grad_out)
            grad_out, grad_w, grad_b = self.layers[i].backprop(grad_out)
            grad_weight[i] = grad_w
            grad_biases[i] = grad_b

        return loss, grad_weight, grad_biases, predictied_probs

    def train(self, x, y, batch_size=50):
        """
        Args:
            x (numpy.ndarray): Input to the neural network, aka training data
            y (numpy.ndarray): Expected output of the neural network, aka training data
        """
        N = x.shape[0]
        num_iters = max(N // batch_size, 1)

        loss_history = []
        for i in range(num_iters):
            batch_indices = np.random.choice(N, batch_size)
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            curr_loss = self._training_step(x_batch, y_batch)
            loss_history.append(curr_loss)
            print "Iteration %d: the current loss is %f" % (i, curr_loss)

        return loss_history


    def _training_step(self, x_batch, y_batch):
        loss, grad_weight, grad_biases, _ = self.loss(x_batch, y_batch)

        for i in grad_weight:
            next_w = self.optimizer.update(self.weights[i], grad_weight[i])
            self.weights[i] = next_w

        for i in grad_biases:
            next_b = self.optimizer.update(self.biases[i], grad_biases[i])
            self.biases[i] = next_b


    def predict(self, input):
        x = input
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x, self.weight[i], self.biases[i])
            # print "dense"
            # print x
            x = self.activations[i].forward(x)
            # print "act"
            # print x
        return x


    def calculate_loss(self, input, expected):
        predicted = self.predict(input)
        return avg_loss(predicted, expected)
