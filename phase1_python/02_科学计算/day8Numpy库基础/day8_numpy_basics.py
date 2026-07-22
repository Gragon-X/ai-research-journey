import numpy as np
#索引与其布尔索引和花式索引
img = np.random.randint(0,256,(10,10,3))
print(f"水平翻转如下:{img[:,::-1,:]}")
print(f"垂直翻转如下:{img[::-1,:,:]}")
mask = img[:,:,0] > 200
img[mask] = [255,255,255]
#相关函数应用
sc = np.random.randint(0,100,(30,5))#创建整数
print(sc)
m = np.sum(sc,axis=1)#对行求和
print(f"每个学生总分与平均分分别为{m}\n{m/5}")
print(f"总分最高的三名为:{np.argsort(m)[-3:]}")#排序
z = np.sum(sc,axis=0)#对列求和
print(f"每门总分，平均分,为{z}\n{z/30}\n")
std = [0,0,0,0,0]
max = [0,0,0,0,0]
min = [0,0,0,0,0]
Pass = [0,0,0,0,0]
i = 0
for z in sc.T:
    Pass[i] = float(z[z>60].size/z.size)
    max[i] = int(np.max(z))
    min[i] = int(np.min(z))
    std[i] = float(np.std(z))
    i+=1 
print(f"每一门最高分，最低分，标准差,及格率分别为{max}\n{min}\n{std}\n{Pass}")
sc[sc <60]= -1
print(sc)
#欧几里得矩阵
points = np.random.randn(5, 2)
s = np.sum(points**2, axis=1, keepdims=True)  
t = s+s.T-2*(points @ points.T)
dist = np.sqrt(t)
print(dist)
#卷积
import numpy as np
n = np.random.randint(0,1000,100)
window = [0.2,0.2,0.2,0.2,0.2]
r = np.convolve(n,window,'valid')
print(r)
#卷积2
from numpy.lib.stride_tricks import sliding_window_view

arr = np.array([1, 3, 5, 7, 9, 11])

windows = sliding_window_view(arr, window_shape=5)
print(windows)
# [[ 1  3  5  7  9]
#  [ 3  5  7  9 11]]
# shape = (2, 5)  → 2个窗口，每个窗口5个元素

avg = windows.mean(axis=1)
print(avg)  # [5.0, 7.0]
