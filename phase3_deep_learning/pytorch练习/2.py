import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    pred = model(X)
    print("预测:\n", pred)
    print("四舍五入:\n", (pred > 0.5).float())