import numpy as np
def gradients(x, y, w, b):
    y_ = w * x + b 
    loss = np.mean((y_ - y) ** 2)
    dw = np.mean(2 * (y_ - y) * x)
    db = np.mean(2 * (y_ - y))
    return[dw , db , loss]