import numpy as np
from collections import defaultdict
from functions import function

settings = {}
settings['n'] = 0                   # dimension of word embeddings
settings['window_size'] = 2         # context window +/- center word
settings['min_count'] = 0           # minimum word count
settings['epochs'] = 5000           # number of training epochs
settings['neg_samp'] = 0            # number of negative words to use during training
settings['learning_rate'] = 0.01    # learning rate
np.random.seed(0)                   # set the seed for reproducibility

corpus = []

class skipgram():
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass

    def generate_training_data(self, settings, corpus):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        self.words_list = sorted(list(word_counts.keys()), reverse = False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        traning_data = []

        for sentence in corpus:
            sent_len = len(sentence)

            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j >= 0 and j <= sent_len - 1:
                        w_context.append(self.word2onehot(sentence[j]))
                traning_data.append([w_target, w_context])

        return traning_data

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))

        return  exp_x / np.sum(exp_x, axis=0)

    def forward(self, x_s):
        y_c_list = []
        for x in x_s:
            h = np.dot(self.w1.T, x)
            u = np.dot(self.w2.T, h)
            y_c = function.softmax(self, u)
            y_c_list.append(y_c)
        return y_c_list, h, u

    def backward(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # by adjusting the result of traning, optimize w1 and w2
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

        pass

    def train(self, traning_data):
        self.w1 = np.random.uniform(0.8, -0.8, (self.v_count, self.n))
        self.w2 = np.random.uniform(0.8, -0.8, (self.n, self.v_count))

        for i in range(self.epochs):

            self.loss = 0

            for w_t, w_c in traning_data:
                y_pred_list, h, u = self.forward(w_c)

                EI = np.sum([np.subtract(y_pred, w_t) for y_pred in y_pred_list], axis=0)

                self.backward(EI, h, w_t)
                for i in range(len(y_pred_list)):
                    self.loss += -np.sum(y_pred_list[i][w_t.index(1)])

        pass

    # input a word, return a vector
    def word_vec(self, word):
        word_index = self.word_index[word]
        v_w = self.w1[word_index]
        return v_w

    # input word, returns top [n] most similar words
    def sim(self, word, top_n):
        vw1 = self.word_vec(word)

        word_similarity = {}
        for i in range(self.v_count):

            w = self.w1[i]
            dot_product = np.dot(vw1, w)
            norm_product = np.linalg.norm(vw1) * np.linalg.norm(w)
            similarity = dot_product / norm_product
            word1 = self.index_word[i]

            if word1 != word:
                word_similarity[word1] = similarity

        sorted_similarity = sorted(word_similarity.items(), key = lambda item: item[1])

        for word, sim in sorted_similarity[:top_n]:
            print(word, sim)

        pass