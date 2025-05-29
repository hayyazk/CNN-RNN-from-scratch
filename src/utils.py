import numpy as np

def embedding(x, embedding_matrix):
    return embedding_matrix[x]

def dense(x, W, b):
    return np.dot(x, W) + b

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))