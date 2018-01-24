# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 数字認識テストデータの取得
digits = datasets.load_digits()

# 数字認識テストデータのデータ形状とデータ種別の表示
print(digits.data.shape, type(digits.data.shape))
# 数字認識テストデータ１件分の表示
print(digits.data[0],digits.data[0].shape)

# 数字認識テストデータ(画像表示用の6×6形式データ)の取得
images = digits.images
# 数字認識テストデータ１件分の表示
print(images[0],images[0].shape)

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
# 数字認識テストデータの正解値の推定
verification = clf.predict(verification_data)
print(verification, correct)

# 数字認識検証用データの取得
predict_data = digits.data[-n*4/10:]
# 数字認識テストデータの正解値の取得
expected = digits.target[-n*4/10:]
# 数字認識テストデータの正解値の推定
predicted = clf.predict(predict_data)

# 正解率の表示
print(metrics.classification_report(expected, predicted))
# 誤認識のマトリックスの表示
print(metrics.confusion_matrix(expected, predicted))

# 数字認識テストデータ(画像表示用の6×6形式データ)の取得
predict_images = images[-n*4/10:]
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(predict_images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess:" + str(predicted[i]))
plt.show()
