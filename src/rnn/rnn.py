import numpy as np

def rnn_unidirectional(x, W_x, W_h, b):
    batch_size, time_steps, input_dim = x.shape
    hidden_dim = W_h.shape[0]
    h = np.zeros((batch_size, hidden_dim))
    
    for t in range(time_steps):
        h = np.tanh(np.dot(x[:, t, :], W_x) + np.dot(h, W_h) + b)
    return h

def rnn_bidirectional(x, W_x_f, W_h_f, b_f, W_x_b, W_h_b, b_b):
    h_forward = rnn_unidirectional(x, W_x_f, W_h_f, b_f)
    h_backward = rnn_unidirectional(x[:, ::-1, :], W_x_b, W_h_b, b_b)
    return np.concatenate([h_forward, h_backward], axis=-1)