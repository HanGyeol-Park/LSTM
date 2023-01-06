import numpy as np

class fucntion:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x)

        return  exp_x / np.sum(exp_x)

    def word2onehot(self, word):
        word_vec =



def word2onehot(self, word):
            word_vec = [0 for i in range(0, self.v_count)]
            word_index = self.word_index[word]
            word_vec[word_index] = 1
            return word_vec