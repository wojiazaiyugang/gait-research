from network.fullconnectnetwork import FullConnectNetwork
import gzip
import pickle
import numpy
from setting import MNIST_DATA_FULL_PATH


class MnistNetwork(FullConnectNetwork):
    def __init__(self):
        self.network_name = "mnist"
        self.layer_sizes = [784, 40, 10]
        self.epochs = 30
        self.learn_rate = 3
        self.mini_batch_size = 10
        super().__init__()

    def load_data(self):
        """
        载入mnist数据。
        train_data是list，包含50000个tuple(x，y)。x是(784,1)的numpy.ndarray，表示图片，y是(10,1)的numpy.ndarray，表示该幅图片的label
        validate_data和test_data都是list，包含10000个tuple(x，y)。x同上，y是个int，表示label
        train_data和validate_data/test_data的格式稍有不同，作者说这种格式最方便，他说最方便就最方便吧
        :return:
        """

        with gzip.open(MNIST_DATA_FULL_PATH, "rb") as file:
            # 这里要加上 encoding = bytes
            train, validate, test = pickle.load(file, encoding="bytes")
        train_input_data = [numpy.reshape(i, (784, 1)) for i in train[0]]
        train_input_data_label = [self.int2vector(i) for i in train[1]]
        train_data = zip(train_input_data, train_input_data_label)
        validate_input_data = [numpy.reshape(i, (784, 1)) for i in validate[0]]
        validate_data = zip(validate_input_data, validate[1])
        test_input_data = [numpy.reshape(i, (784, 1)) for i in test[0]]
        test_data = zip(test_input_data, test[1])
        self.data_doc = "mnist数据集"
        super().load_data()
        return map(list, (train_data, validate_data, test_data))



