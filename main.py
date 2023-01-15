import numpy as np
from functions import function

class LSTM():
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        Wf = np.random.randn(hidden_size, hidden_size)
        Wi = np.random.randn(hidden_size, hidden_size)
        Wo = np.random.randn(hidden_size, hidden_size)
        Wc = np.random.randn(hidden_size, hidden_size)
        Uf = np.random.randn(input_size, hidden_size)
        Ui = np.random.randn(input_size, hidden_size)
        Uo = np.random.randn(input_size, hidden_size)
        Uc = np.random.randn(input_size, hidden_size)
        bf = np.random.randn(hidden_size)
        bi = np.random.randn(hidden_size)
        bo = np.random.randn(hidden_size)
        bc = np.random.randn(hidden_size

        dWf = np.zeros_like(Wf)
        dWi = np.zeros_like(Wi)
        dWo = np.zeros_like(Wo)
        dWc = np.zeros_like(Wc)
        dUf = np.zeros_like(Uf)
        dUi = np.zeros_like(Ui)
        dUo = np.zeros_like(Uo)
        dUc = np.zeros_like(Uc)


        self.params = [Wf, Wi, Wo, Wc,  Uf, Ui, Uo, Uc, bf, bi, bo, bc]
        self.cache = None
        self.grads = [dWf, dWi, dWo, dWc, dUf, dUi, dUo, dUc]

    def forward(self, h_prev, c_prev, x):

        Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc = self.params
        forget = function.sigmoid(np.matmul(Wf, h_prev) + np.matmul(x, Uf) + bf)
        input = function.sigmoid(np.matmul(Wi, h_prev) + np.matmul(x, Ui) + bi)
        output = function.sigmoid(np.matmul(Wo, h_prev) + np.matmul(x, Uo) + bo)

        c_pre = np.tanh(np.matmul(Wc, h_prev) + np.matmul(x, Uc) + bc)
        c = forget * c_prev + input * c_pre
        h = output * np.tanh(c)

        self.cache = (h_prev, c_prev, x, forget, input, output, c_pre, c)
        return c, h

    def backward(self, dc, dh):

        Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc = self.params
        h_prev, c_prev, x, forget, input, output, c_pre, c = self.cache
        dWf, dWi, dWo, dWc, dUf, dUi, dUo, dUc = self.grads

        dc_pre = dc * input
        dc_prev = dc * forget
        di = dc_pre * function.dsigmoid(input)
        df = dc_prev * function.dsigmoid(forget)
        do = dh * function.dsigmoid(output)
        dc = np.tanh(c) * do + dc_pre * (1 - np.power(np.tanh(c), 2))

        dWc += np.matmul(h_prev.T, dc)
        dWo += np.matmul(h_prev.T, do)
        dWi += np.matmul(h_prev.T, di)
        dWf += np.matmul(h_prev.T, df)

        dUc += np.matmul(x.T, dc)
        dUo += np.matmul(x.T, do)
        dUi += np.matmul(x.T, di)
        dUf += np.matmul(x.T, df)

        dh_prev = np.matmul(Wf.T, df) + np.matmul(Wi.T, di) + np.matmul(Wo.T, do) + np.matmul(Wc.T, dc)

        return dc_prev, dh_prev
