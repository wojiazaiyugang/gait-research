"""
imdb电影评论二分类问题
"""
from keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
import numpy


def vectorize_sequences(sequences, dimension=10000):
    """
    将整数序列编码为二进制矩阵，one-hot编码
    :param sequences:
    :param dimension:
    :return:
    """
    result = numpy.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, [sequence]] = 1
    return result


if __name__ == "__main__":
    # 数据预处理
    (train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)
    train_data = vectorize_sequences(train_data)
    validate_data = train_data[:10000]
    train_data = train_data[10000:]
    test_data = vectorize_sequences(test_data)
    train_label = numpy.asarray(train_label).astype("float32")
    validate_label = train_label[:10000]
    train_label = train_label[10000:]
    test_label = numpy.asarray(test_label).astype("float32")
    # 构建模型
    network = models.Sequential()
    network.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    network.add(layers.Dense(16,activation="relu"))
    network.add(layers.Dense(1,activation="sigmoid"))
    network.compile(optimizer=optimizers.RMSprop(),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])
    network.fit(train_data,train_label,epochs=5,batch_size=512,validation_data=(validate_data,validate_label))
    # 评估网络
    # results = network.evaluate(test_data, test_label)
    # 预测数据
    print(network.predict(test_data))
