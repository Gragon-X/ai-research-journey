import numpy as np
import matplotlib.pyplot as plt
def gradients(x, y, w, b):
    y_ = w * x + b 
    loss = np.mean((y_ - y) ** 2)
    dw = np.mean(2 * (y_ - y) * x)
    db = np.mean(2 * (y_ - y))
    return[dw , db , loss]
np.random.rand(42)
x = np.random.rand(100) * 10
y = 3 * x + 2 + np.random.randn(100)
w , b = 0.0 , 0.0
lr = 0.01
for epoch in range(50):
    dw,db,loss = gradients(x, y, w, b)
    w -= lr * dw
    b -= lr * db
    if epoch % 10 == 0:
        print(loss,w,b)
plt.scatter(x,y)
plt.plot(x, w*x + b, color="red")
plt.show()
