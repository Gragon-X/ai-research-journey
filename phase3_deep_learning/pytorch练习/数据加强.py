import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

# ===== 1. 数据（加增强！）=====
# TODO: 训练时：随机旋转±10度 + ToTensor
train_transform = transforms.Compose([
    transforms.RandomRotation(10, fill=0),   # 最大旋转角度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移，最多移 10%
    transforms.ToTensor(),
])

# TODO: 测试时：只转 Tensor，不动手脚
test_transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)


# ===== 2. 模型（和之前一样：Conv + BN + Dropout）=====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 7 * 7)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 3. 训练 =====
epochs = 15
train_losses = []
best_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 验证
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    if acc > best_acc:
        best_acc = acc
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

print(f"\nCNN+BN+Dropout: 98.76% | +数据增强: {best_acc:.2f}%")
