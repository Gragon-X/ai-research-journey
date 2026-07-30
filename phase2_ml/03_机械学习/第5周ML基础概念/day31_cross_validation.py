import numpy as np
data = np.arange(100)
np.random.shuffle(data)
folds = np.array_split(data,5)
for i in range(5):
    val = folds[i]
    train = np.concatenate([folds[j] for j in range(5) if j != i])
    print(f"第{i}折")
    print(f"验证集为{val}")
    print(f"训练集为{train}")