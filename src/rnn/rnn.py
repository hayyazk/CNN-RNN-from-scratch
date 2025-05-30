import numpy as np

def rnn_unidirectional(x, W_x, W_h, b, return_sequences=False):
    batch_size, time_steps, _ = x.shape
    hidden_dim = W_h.shape[0]
    h = np.zeros((batch_size, hidden_dim))

    outputs = []
    
    for t in range(time_steps):
        h = np.tanh(np.dot(x[:, t, :], W_x) + np.dot(h, W_h) + b)
        outputs.append(h)
    
    if return_sequences:
        return np.stack(outputs, axis=1)
    else:
        return h

def rnn_bidirectional(x, W_x_f, W_h_f, b_f, W_x_b, W_h_b, b_b, return_sequences=False):
    h_f = rnn_unidirectional(x, W_x_f, W_h_f, b_f, return_sequences)
    h_b = rnn_unidirectional(x[:, ::-1, :], W_x_b, W_h_b, b_b, return_sequences)

    return np.concatenate([h_f, h_b], axis=-1)