import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_acc, best_params = 0, None
for n in [10, 50, 100]:
    for m in [3, 5, None]:
        model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=42)
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        acc = accuracy_score(y_test, y_pre)
        marcket = ' 🏆' if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_params = (n,m)
        print(f"n_estimators={n} max_depth={m} -> {acc:3f}{marcket}")
print(f"\n最佳n_estuimators={best_params[0]} max_depth={best_params[1]} -> {best_acc:.3f}")