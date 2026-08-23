import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== 1. 加载 + 拆分 =====
transform = transforms.ToTensor()

full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# TODO: 把 60000 拆成 50000 训练 + 10000 验证
train_size = 50000
val_size   = 10000
train_data, val_data = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# ===== 2. 模型（和之前一样）=====
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 3. 训练 + 记录 =====
epochs = 10

# TODO: 创建两个空列表，用来记录每轮的损失
train_losses = []
val_losses   = []

for epoch in range(epochs):
    # --- 训练 ---
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)        # 平均训练损失
    train_losses.append(train_loss)               # 记录

    # --- 验证 ---
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)            # 平均验证损失
    val_losses.append(val_loss)                 # 记录
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ===== 4. 画图 =====
# TODO: 用 matplotlib 画两条线：train_losses（蓝色） vs val_losses（红色）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_losses, 'b-o', label='测试')
plt.plot(range(1, epochs+1), val_losses,   'r-o',    label='验证')
plt.xlabel('轮数')
plt.ylabel('损失')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
