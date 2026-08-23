import torch
w = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([2.0])
y_true = torch.tensor([5.0])

y_pred = w * x
loss = (y_pred - y_true) ** 2

loss.backward()
print("梯度：", w.grad.item())
with torch.no_grad():
    w -= 0.1 * w.grad
print("新w：",w.item())