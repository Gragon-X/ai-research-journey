import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(0, 100, (50, 5)),
                  columns=["语文", "数学", "英语", "物理", "化学"])
print(f"数学大于80的：{df.loc[df["数学"] > 80]}")
df["总分"] = df["语文"] + df["数学"] + df["英语"] + df["物理"] + df["化学"]

df["等级"] = "需努力"
df.loc[df["总分"] >= 300, "等级"] = "良好"
df.loc[df["总分"] >= 400, "等级"] = "优秀"

df = df.sort_values("总分", ascending=False)

print(df.head())
print(df.info())
