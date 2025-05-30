import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, h, w, c = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                x_slice = x[:,
                            i*self.stride:i*self.stride+self.pool_size,
                            j*self.stride:j*self.stride+self.pool_size,
                            :]
                output[:, i, j, :] = np.max(x_slice, axis=(1, 2))
        return output
