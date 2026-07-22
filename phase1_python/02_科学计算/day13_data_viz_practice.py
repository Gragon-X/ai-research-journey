import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 生成数据与处理
班级 = np.random.choice(["一班", "二班", "三班", "四班"], 200)
姓 = ["张","李","王","赵","刘","陈","杨","黄","周","吴"]
名 = ["伟","芳","娜","敏","静","强","磊","洋","婷","杰"]
姓名 = [random.choice(姓) + random.choice(名) for _ in range(200)]

df = pd.DataFrame({
    "班级": 班级,
    "姓名": 姓名,
    "语文": np.random.randint(30, 100, 200),
    "数学": np.random.randint(30, 100, 200),
    "英语": np.random.randint(30, 100, 200),
})
df["总分"] = df[["语文","数学","英语"]].sum(axis=1)
df["等级"] = "需努力"
df.loc[df["总分"] >= 200, "等级"] = "良好"
df.loc[df["总分"] >= 250, "等级"] = "优秀"
avg = df.groupby("班级")[["语文","数学","英语"]].mean()
#子图
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
avg.plot(kind="bar", ax=axes[0])
axes[0].set_title("各科平均分")
df["总分"].plot(kind = "hist", bins = 15, color = "black",ax=axes[1])
axes[1].set_title("分数人数分布")
df["等级"].value_counts().plot(kind="pie",autopct="%.1f%%",ax=axes[2])
axes[2].set_title("等级占比")
axes[3].scatter(df["数学"], df["总分"], alpha=0.5)
axes[3].set_xlabel("数学成绩")
axes[3].set_ylabel("总分")
axes[3].set_title("总分 vs 数学")
plt.show()