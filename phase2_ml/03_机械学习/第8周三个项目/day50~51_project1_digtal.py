import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_digits()
'''
#查看数据基本内容
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head(5))
print(df.describe())
print(data.target_names)
print(data.target)
结果如下：
   pixel_0_0  pixel_0_1  pixel_0_2  pixel_0_3  pixel_0_4  ...  pixel_7_3  pixel_7_4  pixel_7_5  pixel_7_6  pixel_7_7
0        0.0        0.0        5.0       13.0        9.0  ...       13.0       10.0        0.0        0.0        0.0
1        0.0        0.0        0.0       12.0       13.0  ...       11.0       16.0       10.0        0.0        0.0
2        0.0        0.0        0.0        4.0       15.0  ...        3.0       11.0       16.0        9.0        0.0
3        0.0        0.0        7.0       15.0       13.0  ...       13.0       13.0        9.0        0.0        0.0
4        0.0        0.0        0.0        1.0       11.0  ...        2.0       16.0        4.0        0.0        0.0

[5 rows x 64 columns]
       pixel_0_0    pixel_0_1    pixel_0_2    pixel_0_3  ...    pixel_7_4    pixel_7_5    pixel_7_6    pixel_7_7
count     1797.0  1797.000000  1797.000000  1797.000000  ...  1797.000000  1797.000000  1797.000000  1797.000000
mean         0.0     0.303840     5.204786    11.835838  ...    11.809126     6.764051     2.067891     0.364496
std          0.0     0.907192     4.754826     4.248842  ...     4.933947     5.900623     4.090548     1.860122
min          0.0     0.000000     0.000000     0.000000  ...     0.000000     0.000000     0.000000     0.000000
25%          0.0     0.000000     1.000000    10.000000  ...    10.000000     0.000000     0.000000     0.000000
50%          0.0     0.000000     4.000000    13.000000  ...    14.000000     6.000000     0.000000     0.000000
75%          0.0     0.000000     9.000000    15.000000  ...    16.000000    12.000000     2.000000     0.000000
max          0.0     8.000000    16.000000    16.000000  ...    16.000000    16.000000    16.000000    16.000000

[8 rows x 64 columns]
[0 1 2 3 4 5 6 7 8 9]
[0 1 2 ... 8 9 8]

回归任务，无损失，不需要标准化，选择随机森林模型，单次切分，test_size=0.3,预测准确率为0.965
'''
#数据处理
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
for n in [50,100,200]:
    for m in [5, 10, None]:
        model = RandomForestClassifier(n_estimators=n, max_depth=m)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"随机森林(n_estimatorrs={n},max_depth={m})预测手写数字结果：训练准确率={train_acc:.4f}，测试准确率={test_acc:.4f}")
#n=50,m=10最好