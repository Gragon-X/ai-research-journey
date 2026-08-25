import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

#======加载数据=====
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(
    root='xxx',
    train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(
    root='xxx',
    train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

#=====CNN模型====
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
model = CNN()
criterion = nn.CrossEntropyLoss()#损失器
optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
#=======训练========
if os.path.exists('cnn_fashion.pth'):
    model.load_state_dict(torch.load('cnn_fashion.pth'))
    print('已加载模型，跳过训练')
else:
    epochs = 10
    train_losses = []
    best_acc = 0
    for epoch in range(epochs):
    #-------练习--------
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
    #-----测试-----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
    #------计算最佳准确率-----
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    print(f"\nMLP 最高: xxx | CNN 最高: {best_acc:.2f}%")
    torch.save(model.state_dict(), 'cnn_fashion.pth')


#=======可视化=========
model = CNN()
model.load_state_dict(torch.load('cnn_fashion.pth'))
#-------随机取个图（此处凉鞋）------
img_tensor = None
class_names = ['T恤','裤子','毛衣','裙子','外套','凉鞋','衬衫','运动鞋','包','短靴']
for img, label in test_data:
    if  label == 1:        
        img_tensor = img
        break
img_tensor = img_tensor.unsqueeze(0)   # 加批次维度
print(img_tensor.shape)                 # 验证：torch.Size([1, 1, 28, 28])
#-----画conv1的8个卷积核------
fig, axes = plt.subplots(2, 4, figsize=(8,4))
weights = model.conv1.weight.data  # (8, 1, 3, 3)
model.eval()
for i in range(8):
    ax = axes[i // 4][i % 4]
    #取出第 i 个核: (1, 3, 3) → (3, 3)
    kernel = weights[i, 0]
    ax.imshow(kernel, cmap="grey", vmin=-1, vmax=1)
    ax.set_title(f'核{i+1}')
    ax.axis('off')
plt.suptitle(f'凉鞋的 8 个卷积核（数字{label}的训练结果）')
plt.tight_layout()
plt.show()
#------画图片特征图------
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
with torch.no_grad():
    x = model.conv1(img_tensor)
    x = model.relu1(x)
for i in range(8):
    ax = axes[i // 4][i % 4]
    feature_map = x[0, i]
    ax.imshow(feature_map, cmap='hot')
    ax.set_title(f'特征 {i+1}')
    ax.axis('off')
plt.suptitle(f'凉鞋的8 个特征图（输入：数字 {label}）')
plt.tight_layout()
plt.show()

#========核激活统计=======
model = CNN()
model.load_state_dict(torch.load('cnn_fashion.pth'))
model.eval()

pre_sum = torch.zeros(8)
post_sum = torch.zeros(8)
batch = 0

with torch.no_grad():
    for images, labels in test_loader:
        x = model.conv1(images)
        pre_sum += x.mean(dim=(0, 2, 3))
        post_sum += torch.relu(x).mean(dim=(0, 2, 3))
        batch += 1
pre_avg = pre_sum / batch
post_avg = post_sum / batch 

for i in range(8):
    print(f"核{i+1}：RELU前为{pre_avg[i]:.3f},RELU后{post_avg[i]:.3f}")
