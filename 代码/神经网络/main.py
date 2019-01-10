from data.data import load_mnist_data, convent_image_to_mnist_format,load_data0
from network.fullconnectnetwork import FullConnectNetwork
from setting import MODELS_PATH, MNIST_DATA_FULL_PATH, DATA0_PATH
import logging, os, numpy, cv2
from network.mnist import MnistNetwork
from network.walk import WalkNetwork

logging.basicConfig(level=logging.INFO)


def gengerate_full_connect_model(layer_sizes: list, epochs: int, mini_batch_size: int, learn_rate: float):
    """
    生成一个模型
    :return: 模型
    """
    def load_data(data_name:str):
        """
        载入用于训练的数据
        :param data_name: 数据的名字
        :return: train_data, validate_data, test_data
        """
        assert data_name in ["mnist", "data0"], "数据名字错误"
        if data_name == "mnist":
            return load_mnist_data(MNIST_DATA_FULL_PATH)
        elif data_name == "data0":
            return load_data0(DATA0_PATH)

    save_model_full_name = os.path.join(MODELS_PATH,"s" + "-".join(list(map(str, layer_sizes))) + "e" + str(epochs) + "m" + str(
        mini_batch_size) + "l" + str(learn_rate))
    if os.path.isfile(save_model_full_name):
        logging.info("模型 {0} 已经存在".format(save_model_full_name))
        network = FullConnectNetwork()
        network.load_model(save_model_full_name)
    else:
        logging.info("模型 {0} 不存在，开始训练".format(save_model_full_name))
        train_data, validate_data, test_data = load_data("data0")
        network = FullConnectNetwork(layer_sizes)
        network.train(train_data, epochs, mini_batch_size, learn_rate, test_data,
                      save_model_full_name=save_model_full_name)
    return network


if __name__ == "__main__":
    # data0 = load_data0(DATA0_PATH)
    # 得到一个网络
    # train_data, validate_data, test_data = load_data0(DATA0_PATH)
    # network = gengerate_full_connect_model([30, 20, 10], 30, 10, 3)
    # print(network.evaluate(test_data)/len(test_data))
    # train_data, validate_data, test_data = load_mnist_data(MNIST_DATA_PATH)
    # load_mnist_data(MNIST_DATA_FULL_PATH)
    # cv2.imshow("1",numpy.resize(test_data[111][0],(28,28)))
    # cv2.waitKey(0)
    # print(numpy.argmax(network.feedforward(test_data[2][0])),test_data[2][1])
    # print(numpy.argmax(network.feedforward(convent_image_to_mnist_format("./data/1.jpeg"))))
    # for i in range(1, 10):
    #     print(numpy.argmax(network.feedforward(convent_image_to_mnist_format("./data/{0}.png".format(i)))), " ", i)
    # pass
    mnist_network = WalkNetwork()
    mnist_network.train()
    mnist_network.evaluate()