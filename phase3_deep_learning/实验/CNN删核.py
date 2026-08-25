import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层1: 输入1通道 → 输出8通道, kernel_size=3, padding=1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  
        self.relu1 = nn.ReLU()
        #池化层: 2×2 窗口
        self.pool1 = nn.MaxPool2d(2)

        #卷积层2: 输入8通道 → 输出16通道, kernel_size=3, padding=1
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)
        #全连接: 16×7×7 → 10
        #图片经过两次池化: 28→14→7
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.bn1(self.conv1(x))    # → (batch, 8, 28, 28)   padding=1 尺寸不变
        x = self.relu1(x)
        x = self.pool1(x)     # → (batch, 8, 14, 14)   尺寸减半

        x = self.bn2(self.conv2(x))     # → (batch, 16, 14, 14)  padding=1 尺寸不变
        x = self.relu2(x)     
        x = self.pool2(x)     # → (batch, 16, 7, 7)    尺寸再减半

        #展平 → 全连接层
        x = x.view(-1, 784)  # (batch, 16*7*7)
        x = self.dropout(x)
        x = self.fc(x)        # → (batch, 10)
        return x
def test(model):
    """跑一遍测试集，返回准确率(%)"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(
    root='xxx',
    train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(
    root='xxx',
    train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
# ① 加载模型
model = CNN()
model.load_state_dict(torch.load('cnn_fashion.pth'))
# ② 手术前(手术定义：删去沉睡的核)
print(f"手术前精度: {test(model):.2f}%")
# ③ 手术
print(f"核5 bias: {model.conv1.bias[5].item():.4f}")

with torch.no_grad():
    model.conv1.weight[5] = 0
    model.conv1.bias[5] = 0

# ④ 手术后
print(f"手术后精度: {test(model):.2f}%")
# 手术后精度打印完之后（此时 model 已置零）
print(f"置零后核5权重: {model.conv1.weight[5].flatten()}")  # 应该全是0
total_ratio = 0
batch = 0
with torch.no_grad():
    for images, labels in test_loader:
        x = torch.relu(model.conv1(images))
        total_ratio += (x[:, 5] > 0).float().mean().item()
        batch += 1
print(f"置零后核5激活比例: {total_ratio / batch:.4f}")

