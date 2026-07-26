import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#数据
np.random.seed(42)
X = np.random.randn(200,2)
y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]
models = {
    "逻辑回归": LogisticRegression(),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "决策树": DecisionTreeClassifier(max_depth=3)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"{name}：训练={train_acc}，测试={test_acc}")
