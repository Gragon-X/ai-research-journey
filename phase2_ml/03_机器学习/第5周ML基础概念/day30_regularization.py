def L(x):
    loss = (x-3)**2+4*(x**2)
    return loss
def g(x):
    gradient = 2*(x-3)+8*x
    return gradient
w = 0
learning = 0.1
for i in range(20):
    w = w - learning * g(w)
    print(w,L(w))