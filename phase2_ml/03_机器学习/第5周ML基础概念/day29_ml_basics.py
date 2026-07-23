def L(x):
    loss = (x-3)**2
    return loss
def g(x):
    gradient = 2*(x-3)
    return gradient
w = 0
learning = 1.5
for i in range(20):
    w = w - learning * g(w)
    print(w,L(w))