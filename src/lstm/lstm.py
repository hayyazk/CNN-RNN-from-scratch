import numpy as np
from utils import sigmoid

def lstm_cell(x_t, h_prev, c_prev, W_x, W_h, b):
    z = np.dot(x_t, W_x) + np.dot(h_prev, W_h) + b
    
    hidden_size = h_prev.shape[1]
    
    i = sigmoid(z[:, :hidden_size])
    f = sigmoid(z[:, hidden_size:2*hidden_size])
    c_hat = np.tanh(z[:, 2*hidden_size:3*hidden_size])
    o = sigmoid(z[:, 3*hidden_size:])

    c_t = f * c_prev + i * c_hat
    h_t = o * np.tanh(c_t)
    
    return h_t, c_t

def lstm_unidirectional(x, W_x, W_h, b):
    batch_size, time_steps, input_dim = x.shape
    hidden_size = W_h.shape[0]
    
    h = np.zeros((batch_size, hidden_size))
    c = np.zeros((batch_size, hidden_size))

    for t in range(time_steps):
        h, c = lstm_cell(x[:, t, :], h, c, W_x, W_h, b)

    return h

def lstm_bidirectional(x, W_x_f, W_h_f, b_f, W_x_b, W_h_b, b_b):
    h_f = lstm_unidirectional(x, W_x_f, W_h_f, b_f)
    h_b = lstm_unidirectional(x[:, ::-1, :], W_x_b, W_h_b, b_b)
    return np.concatenate([h_f, h_b], axis=1)