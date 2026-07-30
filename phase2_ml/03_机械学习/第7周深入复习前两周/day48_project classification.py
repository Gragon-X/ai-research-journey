import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
#数据准备
data = load_diabetes()
X, y = data.data, data.target
#模型
models = {
    "逻辑回归": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "随机森林": RandomForestRegressor(n_estimators=100)
}
#格式处理
print(f'{"模型":<15} {"R²":<8} {"标准差":<8}')
print('-'*33)
#机械学习
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    score = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    print(f"{name:<15} {score.mean():<8.3} {score.std():<8.3}")