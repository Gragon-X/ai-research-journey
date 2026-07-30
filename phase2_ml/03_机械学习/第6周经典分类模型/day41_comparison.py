import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
#数据准备
data = load_wine()
X, y = data.data, data.target
for seed in [0, 1, 42, 99, 123]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"seed={seed:>3}: 测试准确率={acc:.3f}")
#模型选择
models = {
    "全部猜多数类":DummyClassifier(strategy="most_frequent") ,
    "逻辑回归": LogisticRegression(max_iter=501),
    "KNN(k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN(k=15)": KNeighborsClassifier(n_neighbors=15),
    "决策树": DecisionTreeClassifier(max_depth=3),
    "随机森林": RandomForestClassifier(n_estimators=100),
    "GBoost": GradientBoostingClassifier(n_estimators=100)
}
#训练与成果
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}：训练：{train_acc}，测试：{test_acc}")