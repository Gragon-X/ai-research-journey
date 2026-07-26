import numpy as np
def knn_predict(X_train, y_train, X_test, k = 3):
    predictions = []
    for x in X_test:
        distance = np.linalg.norm(X_train - x, axis=1)
        k_nearnest = y_train[np.argsort(distance)[:k]]
        pred = np.bincount(k_nearnest).argmax()
        predictions.append(pred)
    return predictions
np.random.seed(42)
X = np.random.randn(200, 2)  
y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)
X_train, X_test = X[:150],X[150:]
y_train, y_test = y[:150],y[150:]
print("K值实验：训练准确率 vs 测试准确率")
for k in [1, 3, 5, 7, 10, 15, 20, 30, 50]:
    train_pred = knn_predict(X_train, y_train, X_train, k = k)
    test_pred = knn_predict(X_train, y_train, X_test, k = k)
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    print(f"k={k}：训练准确率：{train_acc}，测试准确率：{test_acc}")