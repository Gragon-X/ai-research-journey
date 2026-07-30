import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#数据处理
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scale = StandardScaler()
X_train_s, X_test_s = scale.fit_transform(X_train), scale.transform(X_test)
#模型准备
models = {
    "逻辑回归": LogisticRegression(max_iter=500),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "决策树": DecisionTreeClassifier(max_depth=5),
    "随机森林": RandomForestClassifier(n_estimators=5),
    "Gboost": GradientBoostingClassifier(n_estimators=100)
}

print(f'{"模型":<33} {"训练":<8} {"测试":<8}')
print('-'*33)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f'{name:<30} {train_acc:<8.3} {test_acc:<8.3}')