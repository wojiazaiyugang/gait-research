# 步态识别算法的LSTM实现

import math
import os
import numpy
import logging
import pickle
from scipy import interpolate
from setting import DATA0_PATH
import matplotlib.pyplot as plt
from keras import layers, activations, models, optimizers, losses, metrics
from keras.utils import to_categorical




def detect_cycle(data):
    def distance(list1, list2):
        assert len(list1) == len(list2), "比较欧式距离时两个向量长度应该相等"
        s = 0
        for i in range(len(list1)):
            s = s + math.pow(list1[i] - list2[i], 2)
        return round(math.sqrt(s), 2)

    reference_length = 50  # 数据点模板长度，在50HZ的数据中，长度为50表示使用1S的模板且模板的位置选在了中间
    dis = []
    count = 0  # 这是用来划分走路周期的，在跟模板比较之后，根据波形的波谷进行划分，实际上是两个波谷才是一个完整的走路周期
    result = []
    temp = []
    x2y2z2 = [i[0] * i[0] + i[1] * i[1] + i[2] * i[2] for i in data]
    for i in range(0, len(x2y2z2) - reference_length):
        dis.append(
            distance(x2y2z2[i:i + reference_length], x2y2z2[len(x2y2z2) // 2:len(x2y2z2) // 2 + reference_length]))
    for i in range(1, len(dis) - 1):
        temp.append(data[i])
        if dis[i] < dis[i - 1] and dis[i] < dis[i + 1]:
            count = (count + 1) % 2
            if count == 0:
                result.append(numpy.array(temp))
                temp = []
    return numpy.array(result)


def chazhi(data, people_index):
    for i, data_i in enumerate(data):
        data_i = numpy.array(data_i)
        x_old, y_old, z_old = data_i[:, 0], data_i[:, 1], data_i[:, 2]
        x = numpy.linspace(0, len(data_i), len(data_i))
        x_index = numpy.linspace(0, len(data_i), POINT_NUMBER_PER_CYCLE)
        new_x = interpolate.interp1d(x, x_old, kind="quadratic")(x_index)
        new_y = interpolate.interp1d(x, y_old, kind="quadratic")(x_index)
        new_z = interpolate.interp1d(x, z_old, kind="quadratic")(x_index)
        temp = []
        for j in range(len(new_x)):
            temp.append((new_x[j], new_y[j], new_z[j]))
        data[i] = numpy.array(temp)
    return numpy.array(data)


def load_data_from_people_i(people_index):
    """
    载入第i个人的数据
    :param people_index:
    :return:(data,label)
    """
    result = []
    logging.info("载入数据 {0}".format(people_index))
    with open(os.path.join(DATA0_PATH, "accData{0}.txt".format(people_index)), "r") as file:
        lines = file.readlines()
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            result.append(numpy.array([x, y, z]))
        result = detect_cycle(result)
        result = chazhi(result, people_index)
    return numpy.array(result), [people_index for i in range(len(result))]


def load_all_people_data():
    """
    载入所有人的数据
    :return: （data，label）
    """
    data0_full_path = os.path.join(DATA0_PATH, "data0")
    if os.path.exists(data0_full_path):
        logging.info("数据已存在")
    else:
        data = []
        label = []
        for i in range(CATEGORY_NUMBER):
            data_i, label_i = load_data_from_people_i(i)
            data.extend(data_i)
            label.extend(label_i)
        with open(data0_full_path, "wb") as file:
            file.write(pickle.dumps((numpy.array(data), numpy.array(label))))
    with open(data0_full_path, "rb") as file:
        result = pickle.loads(file.read())
    return result[0], result[1]


if __name__ == "__main__":
    POINT_NUMBER_PER_CYCLE = 100  # 每个周期内点的个数
    CATEGORY_NUMBER = 10 # 分类个数
    DATA_RATIO = [0.6,0.2,0.2] # 训练、验证、测试集合的比例
    BATCH_SIZE = 64
    EPOCHS = 30

    data, label = load_all_people_data()
    mean = data.mean()
    data = data - mean
    std = data.std()
    data = data / std
    index = numpy.arange(len(data))
    numpy.random.shuffle(index)
    data = data[index]
    label = label[index]
    network_input = models.Input(shape=(POINT_NUMBER_PER_CYCLE,3))
    network = layers.LSTM(64)(network_input)
    network = layers.Dense(10,activation=activations.softmax)(network)
    network = models.Model(inputs=[network_input],outputs=[network])
    network.compile(optimizer=optimizers.RMSprop(lr=0.01), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    network.summary()
    label = to_categorical(label,CATEGORY_NUMBER)
    train_data_number = int(len(data) * DATA_RATIO[0])
    validate_data_number = int(len(data) * DATA_RATIO[1])
    train_data, train_label = data[:train_data_number],label[:train_data_number]
    validate_data, validate_label = data[train_data_number:train_data_number+validate_data_number],label[train_data_number:train_data_number+validate_data_number]
    test_data, test_label = data[train_data_number+validate_data_number:],label[train_data_number+validate_data_number:]
    train_history = network.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data=(validate_data,validate_label))
    evaluate_history = network.evaluate(test_data,test_label,batch_size=BATCH_SIZE)
    logging.info("网络结构:{0}".format(network.get_config()))
    logging.info("模型参数:{0}".format(train_history.params))
    logging.info("训练记录;{0}".format(train_history.history))
    logging.info("测试记录:{0}".format(evaluate_history))

