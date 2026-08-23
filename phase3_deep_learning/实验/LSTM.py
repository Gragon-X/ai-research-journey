# 20 条中文短评（1 = 正面，0 = 负面）
reviews = [
    # 正面
    ("这家店服务态度真好", 1),
    ("味道棒极了下次还来", 1),
    ("上菜速度快分量足", 1),
    ("环境干净整洁很舒服", 1),
    ("招牌菜好吃到流泪", 1),
    ("老板人很热情价格实惠", 1),
    ("外卖包装严实汤没洒", 1),
    ("面条劲道汤头鲜美", 1),
    ("服务周到店员微笑很暖", 1),
    ("性价比超高强烈推荐", 1),
    # 负面
    ("难吃得要命再也不来了", 0),
    ("等了一个小时才上菜", 0),
    ("菜里有头发太恶心了", 0),
    ("价格贵分量还少得可怜", 0),
    ("服务员爱答不理态度差", 0),
    ("肉是馊的吃一口就吐了", 0),
    ("环境吵得要死没法聊天", 0),
    ("汤咸得齁嗓子没法喝", 0),
    ("外卖洒了一袋子全是油", 0),
    ("这家店卫生堪忧不敢再来", 0),
]
word2id = {"<unk>": 0}            
UNK_ID = 0                       
for text, _ in reviews:
    for ch in text:
        if ch not in word2id:
            word2id[ch] = len(word2id)     
new_review = "这家的菜烂透了"
ids = [word2id.get(ch, UNK_ID) for ch in new_review]

import torch
import torch.nn as nn
import torch.optim as optim
embed_dim = 16
embedding = nn.Embedding(len(word2id), embed_dim)
ids = torch.tensor([1, 2, 106, 20, 0, 0, 14])
vecs = embedding(ids)
seqs = [torch.tensor([word2id.get(ch, UNK_ID) for ch in text]) for text, _ in reviews]
padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)   # (20, 最长字数)，短的补 0
labels = torch.tensor([label for _, label in reviews])        # (20,)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out, (h,c) = self.lstm(emb)
        logist = self.fc(h[-1])
        return logist

model = SentimentLSTM(vocab_size=128, embed_dim=16, hidden_dim=16, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    logits = model(padded)
    loss = criterion(logits, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        print(f"Epoch{epoch+1}: Loss={loss.item():.4f}, Acc={acc}")
print(f"Logits={logits}")
print("好与UNK分别为：",model.embedding.weight[9],model.embedding.weight[0])

tests = [
        "好吃不贵",
        "贵不好吃",
        "味道不差"
        ]
model.eval()
with torch.no_grad():
    for t in tests:
        ids = torch.tensor([[word2id.get(ch, UNK_ID) for ch in t]])
        logits = model(ids)
        pred = logits.argmax(dim=1).item()
        print(f"{t} -> logists{logits.tolist()} -> 预测{pred}")
        '''平行版：
        改：
        def forward(self, x):
    emb = self.embedding(x)     # ① 每个字 → 16维向量  (batch, 句子长度, 16)
    pooled = emb.mean(dim=1)    # ② 一句话所有字的向量求平均 → 句子向量 (batch, 16)
    logits = self.fc(pooled)    # ③ 句子向量 → 好坏两个分数 (batch, 2)
    return logits
'''