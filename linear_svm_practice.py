# coding: UTF-8
from sklearn import datasets
from sklearn import svm

# Irisテストデータの取得
iris_data = datasets.load_iris()

# Irisテストデータの表示
print(iris_data.data)
print(iris_data.data.shape)
# Irisテストデータの答えの表示
print(iris_data.target)
print(iris_data.target.shape)

print(iris_data.data[-10:])

# 線形サポートベクターマシーン
clf = svm.LinearSVC()
# サポートベクターマシーンによる訓練
clf.fit(iris_data.data, iris_data.target)

# 検証用データから品質を判定
verification = clf.predict([[3,5,1,1]])
print(verification)
