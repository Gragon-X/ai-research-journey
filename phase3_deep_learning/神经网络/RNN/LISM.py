import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import re
import tarfile
import urllib.request
from collections import Counter

# =====下载并解压 IMDB 数据集=====
data_dir = './data/imdb'
if not os.path.exists(os.path.join(data_dir, 'aclImdb')):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    tar_path = './data/aclImdb_v1.tar.gz'

    os.makedirs(data_dir, exist_ok=True)
    print("正在下载 IMDB 数据集（~80MB）...")
    urllib.request.urlretrieve(url, tar_path)
    print("解压中...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("完成！")
# =====读取文本文件=====
def read_imdb(split):
    """从 aclImdb/{split}/{pos|neg}/ 读取所有 .txt 文件"""
    texts, labels = [], []
    for label_name in ['pos', 'neg']:
        folder = os.path.join(data_dir, 'aclImdb', split, label_name)
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_name == 'pos' else 0)
    return texts, labels

train_texts, train_labels = read_imdb('train')
test_texts,  test_labels  = read_imdb('test')
print(f"训练集: {len(train_texts)} 条 | 测试集: {len(test_texts)} 条")
print(f"示例 ({'正面' if train_labels[0] else '负面'}): {train_texts[0][:150]}...")
# =====分词 + 建词表=====
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()
# 取最常见的 10000 个词
vocab_size = 10000
# 统计词频
all_words = Counter()
for text in train_texts:
    all_words.update(tokenize(text))

word_to_id = {word: i+2 for i, (word, _)
              in enumerate(all_words.most_common(vocab_size))}
word_to_id['<PAD>'] = 0
word_to_id['<UNK>'] = 1

def text_to_ids(text):
    return [word_to_id.get(t, 1) for t in tokenize(text)]

max_len = 500

def pad_trunc(ids):
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))

# =====PyTorch Dataset=====
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = pad_trunc(text_to_ids(self.texts[idx]))
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.float)

train_loader = DataLoader(IMDBDataset(train_texts, train_labels), batch_size=64, shuffle=True)
test_loader  = DataLoader(IMDBDataset(test_texts,  test_labels),  batch_size=64, shuffle=False)

# =====LSTM 模型=====
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+2, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)              
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1]                                 
        x = self.fc(x).squeeze()
        return self.sigmoid(x)

model = SentimentLSTM(vocab_size+2, 128, 128)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# =====训训=====
epochs = 5
# 加梯度裁剪
import torch.nn.utils as nn_utils

for epoch in range(epochs):
    model.train()
    total_loss = total_correct = total_samples = 0

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # 加这一行裁剪梯度，防止爆炸
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += ((outputs > 0.5).float() == labels).sum().item()
        total_samples += labels.size(0)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss/total_samples:.4f} | Train Acc: {100*total_correct/total_samples:.2f}%")
    # 验证（不变）
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            correct += ((outputs > 0.5).float() == labels).sum().item()
            total += labels.size(0)
    print(f"           Test Acc: {100*correct/total:.2f}%")
#预测
model.eval()
fn = 0
fp = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        for i in range(len(labels)):
            if labels[i].item() == 0 and preds[i].item() == 1:
                fp += 1
            if labels[i].item() == 1 and preds[i].item() == 0:
                fn += 1
print(f"假正面（差评→好评）：{fp}篇")
print(f"假好评（好评→差评）：{fn}篇")
print(f"总错误：{fp + fn} / 25000 ({(fp + fn)/250:.2f}%)")
model.eval()
fp_reviews = []
#=======查看数据========
# 只拿索引对应的原文
print(type(test_texts))          # <class 'list'>
print(len(test_texts))           # 25000
print(test_texts[0][:200])       # 第 1 条影评的原文（前 200 字符）
print("=== test_labels 是什么 ===")
print(test_labels[0])            # 0 或 1
for batch_idx, (inputs, labels) in enumerate(test_loader):
    print("batch 0 的 inputs 形状:", inputs.shape)   # (64, 200)
    print("batch 0 的 labels 形状:", labels.shape)   # (64,)
    print("第 1 条的数字序列:", inputs[0][:20], "...")  # 前 20 个词 ID
    break   # 只看第一批

model.eval()
fp_samples = []   # 存 (原文, 置信度)
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)
        preds = (outputs > 0.5).float()

        for i in range(len(labels)):
            if labels[i].item() == 0 and preds[i].item() == 1:
                # TODO: 用 batch_idx 和 i 算出 test_texts 的下标
                idx = batch_idx * 64 + i
                fp_samples.append((test_texts[idx], outputs[i].item()))
# 按置信度排序，打印前 5 篇
fp_samples.sort(key=lambda x: x[1], reverse=True)
for text, conf in fp_samples[:5]:
    print(f"[置信度 {conf:.1%}]")
    print(text)
    print("---")
    # =====词表检查=====
bad_words = ["mediocrity", "confusing", "deadening", "banal", "schmaltzy", "puerile"]
for w in bad_words:
    print(f"{w}: {'在词表' if w in word_to_id else '不在词表(<UNK>)'}")

good_words = ["love", "great", "beautiful", "gorgeous", "fun"]
for w in good_words:
    print(f"{w}: {'在词表' if w in word_to_id else '不在词表(<UNK>)'}")
#统计这些词在训练集里出现多少次
from collections import Counter
for w in ["love", "great", "beautiful", "mediocrity", "confusing", "banal"]:
    print(f"{w}: 训练集出现 {all_words[w]} 次")
