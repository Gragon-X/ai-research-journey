import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams["font.sans-serif"] = ["SimHei"]      # Windows 黑体
plt.rcParams["axes.unicode_minus"] = False          # 解决负号显示为方框
班级 = np.random.choice(["一班","二班","三班"],100)
姓 = ["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "周", "吴"]
名 = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "婷", "杰"]
姓名 = [random.choice(姓) + random.choice(名) for _ in range(100)]
df = pd.DataFrame({
    "班级": 班级,
    "姓名": 姓名,
    "语文": np.random.randint(40,100,100),
    "数学": np.random.randint(40,100,100),
    "英语": np.random.randint(40,100,100)
})
avg = df.groupby("班级")[["语文", "数学", "英语"]].mean()
# 准备三组数据
x1 = np.linspace(-2*np.pi, 2*np.pi, 200)
y_sin = np.sin(x1)
y_cos = np.cos(x1)
x2 = np.random.randint(0, 100, 200)
y2 = 0.8 * x2 + np.random.randn(200) * 5
avg = df.groupby("班级")[["语文", "数学", "英语"]].mean()
# 子图 1×3
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(x1, y_sin, label="sin")
axes[0].plot(x1, y_cos, label="cos")
axes[0].plot(x1, np.ones_like(x1), label="1")
axes[0].legend()
axes[0].set_title("三角函数")
avg.plot(kind="bar", ax=axes[1])   # 用 ax= 指定子图
axes[1].set_title("各科平均分")
axes[2].scatter(x2, y2, alpha=0.5)
axes[2].set_title("散点图")
plt.tight_layout()
plt.show()