import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 生成销售数据
random.seed(42)
np.random.seed(42)

日期 = pd.date_range("2025-01-01", "2025-03-31", freq="D")  # 90天
商品 = ["牛奶", "面包", "可乐", "薯片", "鸡蛋", "苹果", "泡面", "纸巾"]
分类 = {"牛奶":"食品", "面包":"食品", "可乐":"饮料", "薯片":"零食",
        "鸡蛋":"食品", "苹果":"水果", "泡面":"食品", "纸巾":"日用品"}
门店 = ["A店", "B店", "C店"]
data = []
for _ in range(500):
    data.append({
        "日期": np.random.choice(日期),
        "商品": np.random.choice(商品),
        "数量": np.random.randint(1, 10),
        "单价": np.random.choice([5, 8, 12, 15, 20, 25, 30]),
        "门店": np.random.choice(门店),
    })

df = pd.DataFrame(data)
df["分类"] = df["商品"].map(分类)
df["销售额"] = df["数量"] * df["单价"]
#数据浏览
df.info()
print(df.describe())
print(f'总销售额为：{df["销售额"].sum()}')
#各商品销量排名
m = df.groupby("分类")["销售额"].sum()
print(m)
#各门业成绩对比
s = df.groupby("门店").agg({
    "销售额": sum,
    "数量": sum
})
s.plot(kind="bar")
plt.show()
#月度趋势
m = df.groupby(df["日期"].dt.month)["销售额"].sum()
m.plot(kind="line",marker="o")
plt.show()
#分类占比
z = df.groupby("商品")["销售额"].sum()
z.plot(kind="pie",autopct="%.1f%%")
plt.show()
#畅销商品画像
top3 = df.groupby("商品")["销售额"].sum().sort_values(ascending=False).head(3).index
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for i,item in enumerate(top3):
    item_data = df[df["商品"] == item].groupby("门店")["销售额"].sum()
    axes[i].bar(item_data.index,item_data.values)
    axes[i].set_title(item)
    axes[i].set_ylabel("销售额")
plt.tight_layout()
plt.show()