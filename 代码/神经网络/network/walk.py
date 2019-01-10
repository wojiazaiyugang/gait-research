from network.fullconnectnetwork import FullConnectNetwork
import logging
import os
from setting import DATA0_PATH
import math
import pickle
import numpy


class WalkNetwork(FullConnectNetwork):
    def __init__(self):
        self.network_name = "walk"
        self.layer_sizes = [90, 10, 10,10,10]
        self.epochs = 30
        self.learn_rate = 3
        self.mini_batch_size = 10
        super().__init__()

    def load_data(self):
        """
        载入步态识别数据，见mnist的load_data函数
        :return: (train_data,validate_data,test_data)
        """
        data0_full_name = os.path.join(DATA0_PATH, "data0")
        if os.path.isfile(data0_full_name):
            logging.info("data0已经存在")
        else:
            logging.info("data0不存在")
            train_data, validate_date, test_data = [], [], []
            for i in range(10):
                logging.info("正在处理第 {0} 组数据".format(i))
                data_for_people_i = []
                with open(os.path.join(DATA0_PATH, "accData{0}.txt".format(i)), "r") as file:
                    lines = file.readlines()
                    data = [self.get_xyz(i) for i in lines]
                    cycles = self.detect_cycle(data)
                    for cycle in cycles:
                        cycle = self.format_cycle(cycle)
                        if cycle is not None:
                            data_for_people_i.append((cycle, i))
                    # 把每一个人的数据分到三个数据集合中
                    train_data.extend(data_for_people_i[:400])
                    # validate_date.extend()
                    test_data.extend(data_for_people_i[400:])
            # train_data里面的label要转成向量
            for i in range(len(train_data)):
                train_data[i] = (train_data[i][0], self.int2vector(train_data[i][1]))
            with open(os.path.join(DATA0_PATH, "data0"), "wb") as output_file:
                output_file.write(pickle.dumps([train_data, validate_date, test_data]))
        data = pickle.load(open(data0_full_name, "rb"))
        return data[0], data[1], data[2]

    @staticmethod
    def get_xyz(data:str):
        """
        获取xyz三个值
        :param data:
        :return:
        """
        timestamp, x, y, z = [float(i) for i in data.split(" ")]
        return x,y,z

    @staticmethod
    def format_data(data: str):
        """
        将原始加速度传感器数据变成合成加速度
        :param data: time,x,y,z
        :return: sqrt(x*x+y*y+z*z)
        """
        timestamp, x, y, z = [float(i) for i in data.split(" ")]
        return math.sqrt(x * x + y * y + z * z)

    @staticmethod
    def detect_cycle(data:list):
        """
        步态周期检测
        :param data: list
        :return: list[list,list,……] 表示划分的周期
        """
        def distance(list1, list2):
            """
            比较两个向量的欧式距离
            :param list1:
            :param list2:
            :return:
            """
            assert len(list1) == len(list2), "比较欧式距离时两个向量长度应该相等"
            list1 = [math.sqrt(i[0]*i[0] + i[1]*i[1] + i[2]*i[2]) for i in list1]
            list2 = [math.sqrt(i[0] * i[0] + i[1] * i[1] + i[2] * i[2]) for i in list2]
            s = 0
            for i in range(len(list1)):
                s = s + math.pow(list1[i] - list2[i], 2)
            return round(math.sqrt(s), 2)

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

    def format_cycle(self, data: list):
        """
        每一个cycle格式化，因为全连接网络输入的个数是一定的，所以要格式化成长度固定的
        :param data:
        :return:
        """
        number =  self.layer_sizes[0] # 格式化之后的数据个数
        if len(data) >= number//3:
            result = [i[0] for i in data[:number//3]] + [i[1] for i in data[:number//3]] + [i[2] for i in data[:number//3]]
            result = numpy.array(result)
            result = numpy.resize(result, (number, 1))
            return result
        else:
            logging.warning("步态周期数据不足 {0}".format(len(data)))
            return None

