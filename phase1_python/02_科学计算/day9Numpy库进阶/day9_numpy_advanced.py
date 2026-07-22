'''

A * B      # 对应位置相乘（元素乘）# 等价于 np.dot(A, B)
A @ B      # 真正的矩阵乘法（点积）
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)      # 求逆
I = A @ A_inv                  # 验证：A·A⁻¹ ≈ 单位矩阵
arr = np.random.randn(100, 5)  # 100个样本，5个特征
arr = np.random.randn(100, 5)  # 100个样本，5个特征

np.mean(arr)          # 全局均值
np.mean(arr, axis=0)  # 每列均值
np.median(arr, axis=1)        # 中位数（robust，不受异常值影响）
np.var(arr, axis=0)           # 方差
np.std(arr, axis=0)            # 标准差
np.percentile(arr, [25, 50, 75], axis=0)  # 四分位数
np.corrcoef(arr.T)             # 特征间的相关系数矩阵（输入要行=样本，列=特征）

'''
import numpy as np
X = np.random.randn(100, 3)   # 100个样本，3个特征
w_true = np.array([1.5, -2.0, 3.0])
y = X @ w_true + np.random.randn(100) * 0.1  # 加噪声
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print("w_true:", w_true)
print("w_hat :", w_hat)
print("误差:", np.abs(w_true - w_hat))
#相关性分析
x = np.random.randn(100)
c1 = x*0.8+np.random.randn(100)*0.5
c2 = x*0.8+np.random.randn(100)*0.5
c3 = np.random.randn(100)
data = np.column_stack([c1 ,c2, c3])  # (100, 3)
corr = np.corrcoef(data.T)   # 注意：输入的行=变量，列=样本，所以需要转置
print(corr)
#Z-score标准化
x = np.random.randn(100,4)
Z = (x - np.mean(x,axis=0))/ np.std(x,axis=0)
print(Z)