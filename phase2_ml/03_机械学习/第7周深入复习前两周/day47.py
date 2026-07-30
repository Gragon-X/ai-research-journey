import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

data = load_breast_cancer()
X, y = data.data, data.target
#30个特征值
pipe_all = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
score_all = cross_val_score(pipe_all, X, y, cv=5).mean()
#10个特征值
pipe_select = make_pipeline(
    StandardScaler(),
    SelectKBest(score_func=f_classif, k=10),
    LogisticRegression(max_iter=500)
)
score_select = cross_val_score(pipe_select, X, y, cv=5).mean()

print(f"全部30的特征：{score_all:.3f}")
print(f"留5特征：{score_select:.3f}")
#查询选择5项
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(StandardScaler().fit_transform(X), y)
selector_idx = np.argsort(selector.scores_)[-5:][::-1]
print("\n最重要的5个特征:")
for i, idx in enumerate(selector_idx):
    print(f'  {i+1}.{data.feature_names[idx]}(scores={selector.scores_[idx]:.1f}')