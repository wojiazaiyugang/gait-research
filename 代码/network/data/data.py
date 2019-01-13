import gzip
import pickle
import numpy
import cv2
import math
import os
import logging
from setting import DATA0_PATH

def int2vector(i: int) -> numpy.ndarray:
    """
    把一个整数变成(10,1)的向量格式，用于格式化train_data的label
    :param i:
    :return:
    """
    result = numpy.zeros((10, 1))
    result[i] = 1
    return result

def load_mnist_data(minist_data_file_path):
    """
    载入mnist数据。
    train_data是list，包含50000个tuple(x，y)。x是(784,1)的numpy.ndarray，表示图片，y是(10,1)的numpy.ndarray，表示该幅图片的label
    validate_data和test_data都是list，包含10000个tuple(x，y)。x同上，y是个int，表示label
    train_data和validate_data/test_data的格式稍有不同，作者说这种格式最方便，他说最方便就最方便吧
    :return: (train_data,validate_data,test_data)
    """

    with gzip.open(minist_data_file_path, "rb") as file:
        # 这里要加上 encoding = bytes
        train, validate, test = pickle.load(file, encoding="bytes")
    train_input_data = [numpy.reshape(i, (784, 1)) for i in train[0]]
    train_input_data_label = [int2vector(i) for i in train[1]]
    train_data = zip(train_input_data, train_input_data_label)
    validate_input_data = [numpy.reshape(i, (784, 1)) for i in validate[0]]
    validate_data = zip(validate_input_data, validate[1])
    test_input_data = [numpy.reshape(i, (784, 1)) for i in test[0]]
    test_data = zip(test_input_data, test[1])
    return map(list, (train_data, validate_data, test_data))

def convent_image_to_mnist_format(image_full_name):
    """
    把一张图片转为mnist的格式
    :param image_full_name:
    :return:
    """

    def normalize(data):
        """
        归一化，用于把灰度值转换为0-1之间
        :param data:
        :return:
        """
        mx = max(data)
        mn = min(data)
        return numpy.array([(i - mn) / (mx - mn) for i in data])

    image = cv2.imread(image_full_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = numpy.resize(image, (784, 1))
    image = normalize(image)
    return image


def distance(list1, list2):
    """
    比较两个向量的欧式距离
    :param list1:
    :param list2:
    :return:
    """
    assert len(list1) == len(list2), "比较欧式距离时两个向量长度应该相等"
    s = 0
    for i in range(len(list1)):
        s = s + math.pow(list1[i] - list2[i], 2)
    return round(math.sqrt(s), 2)


def detect_cycle(data: list):
    """
    步态周期检测
    :param data: list
    :return: list[list,list,……] 表示划分的周期
    """
    reference_length = 50  # 数据点模板长度，在50HZ的数据中，长度为50表示使用1S的模板且模板的位置选在了中间
    dis = []
    count = 0  # 这是用来划分走路周期的，在跟模板比较之后，根据波形的波谷进行划分，实际上是两个波谷才是一个完整的走路周期
    result = []
    temp = []
    for i in range(0, len(data) - reference_length):
        dis.append(distance(data[i:i + reference_length], data[len(data) // 2:len(data) // 2 + reference_length]))
    for i in range(1, len(dis) - 1):
        temp.append(data[i])
        if dis[i] < dis[i - 1] and dis[i] < dis[i + 1]:
            count = (count + 1) % 2
            if count == 0:
                result.append(temp)
                temp = []
    return result


def load_data0(data0_path):
    """
    把data0处理为网络可以使用的结果，结果序列化存在data0中。一个list，里面三个list分别是train_data, validate_data,test_data,里面10个list，每个list表示一个人的数据，list里面是tuple，第一个元素是数据list，长度未定，第二个元素是label，即是哪个人
    :return:
    """
    def format_data(data: str):
        """
        将原始加速度传感器数据变成合成加速度
        :param data: time,x,y,z
        :return: sqrt(x*x+y*y+z*z)
        """
        timestamp, x, y, z = [float(i) for i in data.split(" ")]
        return math.sqrt(x * x + y * y + z * z)

    def format_cycle(data: list):
        """
        每一个cycle格式化，因为全连接网络输入的个数是一定的，所以要格式化成长度固定的
        :param data:
        :return:
        """
        if len(data) > 30:
            result = numpy.array(data[:30])
            result = numpy.resize(result,(30,1))
            return result
        else:
            logging.warning("步态周期数据不足 {0}".format(len(data)))
            return None

    data0_full_name = os.path.join(data0_path, "data0")
    if os.path.isfile(data0_full_name):
        logging.info("data0已经存在")
    else:
        logging.info("data0不存在")
        train_data, validate_date, test_data = [], [], []
        for i in range(10):
            logging.info("正在处理第 {0} 组数据".format(i))
            data_for_people_i = []
            with open(os.path.join(data0_path,"accData{0}.txt".format(i)), "r") as file:
                lines = file.readlines()
                data = [format_data(i) for i in lines]
                cycles = detect_cycle(data)
                for cycle in cycles:
                    cycle = format_cycle(cycle)
                    if cycle is not None:
                        data_for_people_i.append((cycle, i))
                # 把每一个人的数据分到三个数据集合中
                train_data.extend(data_for_people_i[:400])
                # validate_date.extend()
                test_data.extend(data_for_people_i[400:])
        # train_data里面的label要转成向量
        for i in range(len(train_data)):
            train_data[i] = (train_data[i][0],int2vector(train_data[i][1]))
        with open(os.path.join(data0_path,"data0"), "wb") as output_file:
            output_file.write(pickle.dumps([train_data,validate_date,test_data]))
    data = pickle.load(open(data0_full_name,"rb"))
    return data[0], data[1], data[2]


if __name__ == "__main__":
    pass
