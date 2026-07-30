import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)
modles = LogisticRegression(max_iter=500)
modles.fit(X_train_s, y_train)
y_prob = modles.predict_proba(X_test_s)[:, 1]
threshold = float(input())
y_pred = (y_prob >= threshold).astype(int)
f1 = 2*precision_score(y_test, y_pred)*recall_score(y_test, y_pred) / (recall_score(y_test, y_pred) + precision_score(y_test, y_pred))
print(f"阈值 = {threshold}")
print(f"精确率: {precision_score(y_test, y_pred):.3f}")
print(f"召回率: {recall_score(y_test, y_pred):.3f}")
print(f"F1:{f1}")