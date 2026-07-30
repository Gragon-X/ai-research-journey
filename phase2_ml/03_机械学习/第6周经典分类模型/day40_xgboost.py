import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
np.random.seed(42)
X = np.random.randn(500,10)
y = (X[:, 0] + X[:, 2] + X[:, 5] + np.random.randn(500) * 0.3 > 0).astype(int)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
modles = {
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "GBoost": GradientBoostingClassifier(n_estimators=100,random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42)
}
for name,modle in modles.items():
    modle.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, modle.predict(X_train))
    test_acc = accuracy_score(y_test, modle.predict(X_test))
    print(f"{name}：训练={train_acc},测试={test_acc}")