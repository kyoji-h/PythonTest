# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math

# ネイピア数取得
e = math.e

def get_sigmoid(x):
    s = 1 / (1 + e**-x)
    return s

# -8～8の0.1刻みの配列作成
dx = 0.1
sigmoid_arr = np.arange(-8, 8, dx)

#y_sigmoid = 1 / (1 + e**-sigmoid_arr)
#print(y_sigmoid)
# 作成した配列をシグモイド関数にかける
y_sig = get_sigmoid(sigmoid_arr)
# 作成した配列のシグモイド関数の傾き
y_dsig = (get_sigmoid(sigmoid_arr + dx) - get_sigmoid(sigmoid_arr)) / dx

# 図の表示
plt.plot(sigmoid_arr, y_sig)
plt.plot(sigmoid_arr, y_dsig)
plt.show()
