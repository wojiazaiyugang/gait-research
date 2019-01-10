import numpy
import random
import pickle
import os
import logging
from setting import MODELS_PATH


class FullConnectNetwork():
    def __init__(self):
        """
        初始化函数
        """
        # 继承类需要声明的
        self.network_name = self.network_name
        self.layer_sizes = self.layer_sizes
        self.epochs = self.epochs
        self.learn_rate = self.learn_rate
        self.mini_batch_size = self.mini_batch_size
        # 继承类无需声明
        self.train_data,self.validate_data,self.test_data = self.load_data()
        self.model_full_name = os.path.join(MODELS_PATH, self.network_name + "_" + "-".join(
            list(map(str, self.layer_sizes))) + "E" + str(
            self.epochs) + "M" + str(self.mini_batch_size) + "L" + str(self.learn_rate))
        self.biases = [numpy.random.randn(i, 1) for i in self.layer_sizes[1:]]
        self.weights = [numpy.random.randn(j, i) for i, j in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def load_model(self, model_file_full_name):
        """
        从一个文件中加载模型的w和b
        :param model_file_full_name:
        :return:
        """
        with open(model_file_full_name, "rb") as model_file:
            model = pickle.load(model_file)
        self.weights, self.biases = model[0], model[1]

    def train(self):
        """
        随机梯度下降法。如果能找到模型就直接用，找不到就当场训练一个
        :return:
        """
        if os.path.isfile(self.model_full_name):
            logging.info("模型 {0} 已经存在，加载模型".format(self.model_full_name))
            with open(self.model_full_name, "rb") as model_file:
                model = pickle.load(model_file)
            self.weights, self.biases = model[0], model[1]
        else:
            logging.info("模型 {0} 不存在，开始训练".format(self.model_full_name))
            for i in range(self.epochs):
                random.shuffle(self.train_data)
                mini_batchs = [self.train_data[j:j +self.mini_batch_size] for j in range(0, len(self.train_data), self.mini_batch_size)]
                for mini_batch in mini_batchs:
                    self.update_mini_batch(mini_batch, self.learn_rate)
                self.evaluate()
            # 训练之后把模型进行保存
            logging.info("保存模型 {0}".format(self.model_full_name))
            with open(self.model_full_name, "wb") as model_file:
                model_file.write(pickle.dumps([self.weights, self.biases]))
            logging.info("保存模型成功")

    def update_mini_batch(self, mini_batch: list, learn_rate: float):
        """
        通过反向传播更新网络的w和b
        :param mini_batch:
        :param learn_rate:
        :return:
        """
        # nabla表示哈密顿算子，nabla_b表示损失函数对b的变化率
        # 初始化
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for input_data, input_data_label in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(input_data, input_data_label)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learn_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learn_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, input_data, input_data_label):
        """
        反向传播算法。返回损失函数对w和b的偏导数
        :param input_data:
        :param input_data_label:
        :return: (nabla_b, nabla_w)
        """
        # 初始化
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # 前向传播
        activation = input_data  # 上一层的输出
        activations = [input_data]  # 所有每一层的输出
        zs = []  # 所有的激活值，输出 = sigmoid(激活值)
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # 反向传播
        delta = (activations[-1] - input_data_label) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())  # TODO 这里为啥转置的数学原理不知道
        for i in range(2, len(self.layer_sizes)):
            z = zs[-i]
            sp = self.sigmoid_prime(z)
            delta = numpy.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = numpy.dot(delta, activations[-i - 1].transpose())
        return nabla_b, nabla_w

    def sigmoid(self, z):
        """
        sigmoid函数，activation = sigmoid(z)
        这里对sigmoid函数进行优化，否则会出现 RuntimeWarning: overflow encountered in exp
        :param z: 
        :return: 
        """
        # old
        # return 1 / (1 + numpy.exp(-z))

        # new
        result = []
        for i in z:
            if i >= 0:
                result.append(1 / (1 + numpy.exp(-i)))
            else:
                result.append(numpy.exp(i)/(1+numpy.exp(i)))
        return numpy.resize(numpy.array(result),z.shape)

    def sigmoid_prime(self, z):
        """
        TODO sigmoid函数的导数，为啥用这个玩意不知道
        :param z:
        :return:
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def evaluate(self):
        """
        评估网络准确率
        :return:
        """
        test_results = [(numpy.argmax(self.feedforward(input_data)), input_data_label) for
                        (input_data, input_data_label) in self.test_data]
        result = sum(int(output_result == input_data_label) for (output_result, input_data_label) in test_results)
        logging.info("{0:>5}/{1:<5}准确率{2:>6.2f}%".format(result, len(self.test_data),100*result/len(self.test_data)))
        return result

    def feedforward(self, input_data):
        """
        给定一个输入样例然后正向传播得到预测值
        :param input_data:
        :return:
        """
        result = input_data
        for b, w in zip(self.biases, self.weights):
            result = self.sigmoid(numpy.dot(w, result) + b)
        return result

    def load_data(self):
        """
        初始化网络的时候加载数据
        :return:(train_data,validate_data,test_data)
        """
        logging.info("数据说明:{0}".format(self.data_doc))
        return None,None,None

    @staticmethod
    def int2vector(i: int) -> numpy.ndarray:
        """
        把一个整数变成(10,1)的向量格式，用于格式化train_data的label
        :param i:
        :return:
        """
        result = numpy.zeros((10, 1))
        result[i] = 1
        return result
