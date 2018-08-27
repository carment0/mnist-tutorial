import numpy as np
from implementation.models import avg_loss
from gradient_check import eval_numerical_gradient


class Softmax(object):
    def __init__(self):
        self.output = None

    def numerical_stable_probability(self, x):
        """Computes softmax probabilities using numerically stable technique"""
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

        return np.exp(shifted_logits) / Z

    def forward(self, scores):
        probs = self.numerical_stable_probability(scores)
        self.output = probs

        return probs

    def backprop(self, y):
        probs = self.output

        if probs is None:
            raise "please perform forward prop before back prop"

        # Fact: scores, grad_scores, and probs all have the same shape
        grad_scores = np.zeros_like(probs)
        itr = np.nditer(grad_scores, flags=['multi_index'], op_flags=['readwrite'])
        while not itr.finished:
            i, k = itr.multi_index
            if y[i] == k:
                grad_scores[i][k] = probs[i][k] - 1
            else:
                grad_scores[i][k] = probs[i][k]
            itr.iternext()

        return grad_scores / len(probs)
        # for i in range(len(scores)):
        #     for j in range(len(scores[i])):
        #         init_val = scores[i][j]
        #         scores[i][j] = init_val + delta
        #         f_plus = avg_loss(self.forward(scores), y)
        #         scores[i][j] = init_val - delta
                # f_minus = avg_loss(self.forward(scores), y)
        #         scores[i][j] = init_val
        #         grad_score[i][j] = (f_plus - f_minus) / (2 * delta)

        # return grad_score


# N x C => number of input, catagories
# if __name__ == '__main__':
#     dummy = Softmax()
#     arr = [100, 90, 70, -100]
#     scores = np.array(arr)
#     print dummy.calculate_probability(scores, 0)
#     # 0.999955..

#     print dummy.forward(scores)
#     #[9.99954602e-01, 4.53978687e-05, 9.35719815e-14, 1.38383370e-87]
#     # aka array of classes with probabilities => prediction

