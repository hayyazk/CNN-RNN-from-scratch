import numpy as np

def embedding(x, embedding_matrix):
    return embedding_matrix[x]

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

def dense(x, W, b):
    return np.dot(x, W) + b

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)