import numpy as np

class Conv2D:
    def __init__(self, W, b, stride=1, padding='same'):
        self.W = W  # shape: (kh, kw, in_channels, out_channels)
        self.b = b
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        batch_size, in_h, in_w, in_c = x.shape
        kh, kw, _, out_c = self.W.shape
        
        if self.padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
        else:
            pad_h = pad_w = 0

        out_h = (in_h + 2 * pad_h - kh) // self.stride + 1
        out_w = (in_w + 2 * pad_w - kw) // self.stride + 1

        x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

        output = np.zeros((batch_size, out_h, out_w, out_c))

        for b in range(batch_size):
            for oc in range(out_c):
                for i in range(0, out_h):
                    for j in range(0, out_w):
                        for ic in range(in_c):
                            region = x_padded[b,
                                              i * self.stride:i * self.stride + kh,
                                              j * self.stride:j * self.stride + kw,
                                                ic]
                            output[b, i, j, oc] += np.sum(region * self.W[:, :, ic, oc])
                output[b, :, :, oc] += self.b[oc]
        
        return output
