import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

data = load_diabetes()

X, y= data.data, data.target
score = cross_val_score(LinearRegression(), X, y, cv=5)
print(score)
print(score.mean())