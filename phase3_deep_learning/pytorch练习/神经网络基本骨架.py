import torch
import torch.nn as nn
import torch.optim as optim
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义网络：1个输入 → 1个输出（就是 w*x，连偏置都不要）
model = nn.Linear(1, 1, bias=False)

# 手动设置权重为 1（模拟你之前的 w=1）
with torch.no_grad():
    model.weight.fill_(1.0)

# 2. 定义损失函数和优化器
criterion = nn.MSELoss()       # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 优化器

# 3. 训练一步
x = torch.tensor([[2.0]])       # 输入（注意：必须二维）
y_true = torch.tensor([[5.0]])  # 真值

y_pred = model(x)               # 正向传播
loss = criterion(y_pred, y_true) # 算损失

optimizer.zero_grad()           # 清空旧梯度
loss.backward()                 # 反向传播
optimizer.step()                # 更新参数

print(f"权重: {model.weight.item():.4f}")   # 应该 ≈ 2.2
print(f"预测: {y_pred.item():.4f}")          # 应该是 2.0
print(f"损失: {loss.item():.4f}")             # 应该是 9.0
'''
x = torch.tensor([2.0])
y_true = torch.tensor([5.0])

model = nn.Linear(1, 1, bias=True)
with torch.no_grad():
    model.weight.fill_(1.0)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Epoch {epoch+1:2d} | w={model.weight.item():.4f} | pred={y_pred.item():.4f} | loss={loss.item():.6f}")