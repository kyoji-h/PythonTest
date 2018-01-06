# coding: utf-8
import math

# シグモイド関数
def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))

# ニューロンクラス
class Neuron:
    # ニューロンへの入力値
    input_sum = 0.0
    output = 0.0

    # ニューロンへの入力値加算処理
    def setInput(self, value):
        self.input_sum += value

    # ニューロンからの出力値取得処理
    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        #self.output = self.input_sum
        return self.output

# ニューラルネットワーククラス
class NeuralNetwork:
    neuron = Neuron()

    def commit(self, input_data_list):
        for data in input_data_list:
            self.neuron.setInput(data)
        return self.neuron.getOutput()

neural_network = NeuralNetwork()
result = neural_network.commit([2.0, 3.0, 6,0])
print(result)