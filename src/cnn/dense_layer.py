class Flatten:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return x @ self.W + self.b
