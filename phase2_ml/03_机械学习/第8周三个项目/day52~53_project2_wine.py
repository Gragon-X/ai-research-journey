import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
data = load_wine()
'''
#查看数据
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head(5))
print(df.describe())
print(data.target_names)
print(data.target)
结果如下：
   alcohol  malic_acid   ash  alcalinity_of_ash  ...  color_intensity   hue  od280/od315_of_diluted_wines  proline
0    14.23        1.71  2.43               15.6  ...             5.64  1.04                          3.92   1065.0
1    13.20        1.78  2.14               11.2  ...             4.38  1.05                          3.40   1050.0
2    13.16        2.36  2.67               18.6  ...             5.68  1.03                          3.17   1185.0
3    14.37        1.95  2.50               16.8  ...             7.80  0.86                          3.45   1480.0
4    13.24        2.59  2.87               21.0  ...             4.32  1.04                          2.93    735.0

[5 rows x 13 columns]
          alcohol  malic_acid         ash  ...         hue  od280/od315_of_diluted_wines      proline
count  178.000000  178.000000  178.000000  ...  178.000000                    178.000000   178.000000
mean    13.000618    2.336348    2.366517  ...    0.957449                      2.611685   746.893258
std      0.811827    1.117146    0.274344  ...    0.228572                      0.709990   314.907474
min     11.030000    0.740000    1.360000  ...    0.480000                      1.270000   278.000000
25%     12.362500    1.602500    2.210000  ...    0.782500                      1.937500   500.500000
50%     13.050000    1.865000    2.360000  ...    0.965000                      2.780000   673.500000
75%     13.677500    3.082500    2.557500  ...    1.120000                      3.170000   985.000000
max     14.830000    5.800000    3.230000  ...    1.710000                      4.000000  1680.000000


[8 rows x 13 columns]
['class_0' 'class_1' 'class_2']
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 '''
#数据处理
X, y =data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
#随机森林切片
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f'随机森林切片评估：\n训练={train_acc},测试={test_acc}')
#KNN切片
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"KNN(K=5)时，训练={train_acc}，测试={test_acc}")
    #砍掉一半特征
pipe = make_pipeline(
    SelectKBest(score_func=f_classif, k=6),
    RandomForestClassifier(n_estimators=50,max_depth=6)
)
score6 = cross_val_score(pipe, X, y, cv=5).mean()
print(f'随机森林（6个特征）交叉验证：{score6}')
    #对比13个特征
score_all = cross_val_score(RandomForestClassifier(n_estimators=50, max_depth=10), X, y, cv=5).mean()
print(f'随机森林（13个特征）交叉验证：{score_all}')