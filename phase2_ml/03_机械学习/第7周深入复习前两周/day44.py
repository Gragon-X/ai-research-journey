import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = load_breast_cancer()
X, y = data.data, data.target

print("单词切分（三次不同的random_state）:")
for rs in [0, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)
    scaler = StandardScaler()
    modle = LogisticRegression(max_iter=500)
    modle.fit(scaler.fit_transform(X_train), y_train)
    acc = modle.score(scaler.transform(X_test), y_test)
    print(f"random_state={rs},测试准确率={acc:.3f}")

print()
print("5折交叉验证:")
pipe = make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
scores = cross_val_score(pipe, X, y, cv=5)
print(f"每折得分为：{np.round(scores,3)}")
print(f"平均+-标准差：{scores.mean():.3f}+-{scores.std():.3f}")
# 对比不同正则化强度
for C_val in [0.01, 0.1, 1.0, 10, 100]:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C_val, max_iter=500))
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f'C={C_val:<5}  交叉验证准确率={scores.mean():.3f}')