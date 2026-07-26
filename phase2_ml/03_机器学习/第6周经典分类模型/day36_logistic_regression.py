import numpy as np
def sigmoid(z):
    return 1 / (1 + np.e ** -z)
def forward(x, w, b):
    return sigmoid(w*x + b)
def compute_loss(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
def gradient(x, y_pred, y_true):
    dw = np.mean((y_pred - y_true)[:, None] * X, axis=0)
    db = np.mean(y_pred - y_true)
    return [dw,db]
# 合成数据（两个簇）
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2)  # 2个特征
# 真决策边界：x1 + x2 > 0 → 正类
y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
lr = 0.1
w , b = np.zeros(2) , 0.0
for epochs in range(1000):
    z = X @ w + b
    y_pred = sigmoid(z)
    dw,db = gradient(X, y_pred , y)
    w -= lr * dw
    b -= lr * db
    loss = compute_loss(y_pred , y)
    if epochs % 100 == 0:
        print(loss)
