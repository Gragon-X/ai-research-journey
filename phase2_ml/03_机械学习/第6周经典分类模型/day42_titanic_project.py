import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 加载数据（本地，不用网络）
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print(f'数据集: 乳腺癌诊断 (二分类)')
print(f'样本数: {X.shape[0]}, 特征数: {X.shape[1]}')
print(f'类别: 恶性={ (y==1).sum() }, 良性={ (y==0).sum() }')
print()

# 切分 + 缩放
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    '逻辑回归': LogisticRegression(max_iter=500),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
    '决策树': DecisionTreeClassifier(max_depth=5),
    '随机森林': RandomForestClassifier(n_estimators=100),
    'GBoost': GradientBoostingClassifier(n_estimators=100),
}

print(f'{"模型":<15} {"训练":<8} {"测试":<8}')
print('-' * 33)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f'{name:<15} {train_acc:<8.3f} {test_acc:<8.3f}')

# 特征重要性
rf = models['随机森林']
print('\n最重要的5个特征:')
importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
for name, imp in importances[:5]:
    print(f'  {name:<30} {imp:.3f}')
