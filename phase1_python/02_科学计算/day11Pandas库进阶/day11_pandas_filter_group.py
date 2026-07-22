import numpy as np
import pandas as pd
import random
# 班级
班级 = np.random.choice(["一班", "二班", "三班"], 100)
# 姓名（随机组合）
姓 = ["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "周", "吴"]
名 = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "婷", "杰"]
姓名 = [random.choice(姓) + random.choice(名) for _ in range(100)]
# 拼成 DataFrame
df = pd.DataFrame({
    "班级": 班级,
    "姓名": 姓名,
    "语文": np.random.randint(40, 100, 100),
    "数学": np.random.randint(40, 100, 100),
    "英语": np.random.randint(40, 100, 100),
})
df["总分"] = df["语文"] + df["数学"] + df["英语"]
print("每班平均分分别为：")
s = df.groupby("班级")[["语文","数学","英语"]].mean()
print(s)
m = df.groupby("班级").agg({
   "数学": ["max","min","mean","count"]
})
print(m)
sa = df[df["数学"]>=60]
a = sa.groupby("班级")["数学"].count()
print("每班及格人数如下：")
print(a)
print(df)
df["等级"] = "低"
df.loc[df["总分"] >= 200,"等级"] = "中"
df.loc[df["总分"] >= 270,"等级"] = "高"
r = df.groupby(["班级", "等级"]).size()
print(r)