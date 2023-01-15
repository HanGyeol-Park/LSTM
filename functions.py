import numpy as np

class function:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))

        return  exp_x / np.sum(exp_x, axis=0)

    def dsigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
