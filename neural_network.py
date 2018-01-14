# coding: UTF-8
import math
import matplotlib.pyplot as plt

# シグモイド関数
def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))

# ニューロンクラス
class Neuron:
    # ニューロンへの入力値と出力値
    input_sum = 0.0
    output = 0.0

    # ニューロンへの入力値加算処理
    def setInput(self, value):
        self.input_sum += value

    def getInput(self):
        return self.input_sum

    # ニューロンからの出力値取得処理
    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0.0
        self.output = 0.0

# ニューラルネットワーククラス
class NeuralNetwork:
    # 入力層：入力値としてcommitメソッドのパラメータが設定される（ここではデフォルト値）
    input_layer = [0.0, 0.0, 1.0]
    # 中間層：ニューロンクラス×２とバイアス（固定値）
    middle_layer = [Neuron(), Neuron(), 1.0]
    # 出力層：ニューロンクラス
    output_layer = Neuron()

    # 入力層から中間層１、および入力層から中間層２への重み付け
    w_im_1 = [0.496, -0.501, 0.498]
    w_im_2 = [0.512, 0.998, -0.502]
    # 中間層から出力層への重み付け
    w_mo = [0.121, -0.4996, 0.200]

    def getMiddleLayerDetail(self):
        middle_input = [self.middle_layer[0].getInput(), self.middle_layer[1].getInput(), self.middle_layer[2]]
        middle_output = [self.middle_layer[0].getOutput(), self.middle_layer[1].getOutput(), self.middle_layer[2]]
        return {'middle_input':middle_input, 'middle_output':middle_output}

    def getOutputLayerDetail(self):
        o_layer_input = self.output_layer.getInput()
        o_layer_output = self.output_layer.getOutput()
        return {'o_layer_input':o_layer_input, 'o_layer_output':o_layer_output}

    def commit(self, input_data_list):
        # 入力層への設定
        self.input_layer[0] = input_data_list[0]
        self.input_layer[1] = input_data_list[1]

        # ニューロンのリセット
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.output_layer.reset()

        print(self.input_layer)

        cnt = 0
        for input_data in self.input_layer:
            # 中間層１への重み付け取得
            w_1 = dict(enumerate(self.w_im_1)).get(cnt, -1)
            # 中間層２への重み付け取得
            w_2 = dict(enumerate(self.w_im_2)).get(cnt, -1)
            if not w_1 == -1:
                # 中間層１のニューロンへ入力値×重みを設定
                self.middle_layer[0].setInput(input_data * w_1)
            if not w_2 == -1:
                # 中間層２のニューロンへ入力値×重みを設定
                self.middle_layer[1].setInput(input_data * w_2)
            cnt = cnt + 1

        print(self.getMiddleLayerDetail())

        cnt = 0
        for middle_data in self.middle_layer:
            # 出力層への重み付け取得
            w = dict(enumerate(self.w_mo)).get(cnt, -1)
            if not w == -1:
                # 出力層のニューロンへ中間層入力値×重みを設定
                if isinstance(middle_data, Neuron):
                    self.output_layer.setInput(middle_data.getOutput() * w)
                else:
                    self.output_layer.setInput(middle_data * w)

        print(self.getOutputLayerDetail())

        return self.output_layer.getOutput()



base_latitude = 34.5
base_longitude = 137.5

# データファイルの読込
trial_data = []
file = open('trial-data.txt', 'r')
for line in file:
    line_arr = line.rstrip().split(',')
    trial_data.append([float(line_arr[0]) - base_latitude, float(line_arr[1]) - base_longitude])
file.close()

# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()

output = neural_network.commit([3,5])
print(output)
