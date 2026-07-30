import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_digits()
X, y = data.data, data.target
print(f"类别为{data.target_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "逻辑回归":LogisticRegression(max_iter=500),
    "KNN(k=5)":KNeighborsClassifier(n_neighbors=5),
    "决策树":DecisionTreeClassifier(max_depth=3),
    "随机森林":RandomForestClassifier(n_estimators=50)
}
print(f'{"模型":<15} {"训练":<8} {"测试":<8}')
print('-' * 33)
for name,model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'{name:<15} {train_acc:<8.3} {test_acc:<8.3}')