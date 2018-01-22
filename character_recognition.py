# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 数字認識テストデータの取得
digits = datasets.load_digits()

print(digits.data.shape, type(digits.data.shape))

images = digits.images

# 数字認識テストデータの総数取得
n = len(digits.data)

# サポートベクターマシーン
clf = svm.SVC(gamma=0.001, C=1000.0)
# サポートベクターマシーンによる訓練
clf.fit(digits.data[:n*6/10],digits.target[:n*6/10])

# 数字認識検証用データの取得
verification_data = digits.data[-10:]
# 数字認識テストデータの正解値の取得
correct = digits.target[-10:]
# 数字認識テストデータの判定
verification = clf.predict(verification_data)
print(verification, correct)