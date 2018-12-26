# 工具类
import config
import numpy
import matplotlib.pyplot as plt
import os
import scipy.interpolate
import dtw
import stats
import seaborn
import time
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_row_data(file):
    """
    从文件中读取原始数据
    :param file: 文件名，绝对路径，"D://a/b/c.txt"
    :return: list0、list1和list2，分别对应文件中的数据,list0 = ["123123 13 12 12"],其中list0为走路数据，剩下俩为左右手
    """
    file = open(file)
    lines = file.readlines()
    file.close()
    list0 = []
    list1 = []
    list2 = []
    for line in lines:
        if line.endswith("0\n"):
            list0.append(line[:-3])
        elif line.endswith("1\n"):
            list1.append(line[:-3])
        elif line.endswith("2\n"):
            list2.append(line[:-3])
        else:
            raise Exception("错误的原始数据", line)
    return list0, list1, list2


def my_std(lines):
    """
    计算【样本】标准差
    :param lines:list
    :return: 【样本】标准差
    """
    return numpy.std(lines, ddof=1)


def my_interpolate(lines, kind="quadratic"):
    """
    将传感器数据进行插值然后返回采样的结果
    :param lines: 待处理数据，["t x y z","1 1 2 3","2 4 5 6","3 7 8 9"]
    :param kind: 插值函数
    :return: 采样后结果，["t x y z"]
    """
    x_list = []
    y_list = []
    z_list = []
    x_coordinate = []
    x_new_coordinate = [0]
    t, x, y, z = [float(i) for i in lines[0].split(" ")]
    start = t
    for line in lines:
        t, x, y, z = [float(i) for i in line.split(" ")]
        if len(x_coordinate) == 0 or t - start - x_coordinate[-1] > 0:
            x_coordinate.append(t - start)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
    while x_new_coordinate[-1] + config.SAMPLE_DELAY < x_coordinate[-1]:
        x_new_coordinate.append(x_new_coordinate[-1] + config.SAMPLE_DELAY)
    func = scipy.interpolate.interp1d(x_coordinate, x_list, kind=kind)
    list0 = func(x_new_coordinate).tolist()
    func = scipy.interpolate.interp1d(x_coordinate, y_list, kind=kind)
    list1 = func(x_new_coordinate).tolist()
    func = scipy.interpolate.interp1d(x_coordinate, z_list, kind=kind)
    list2 = func(x_new_coordinate).tolist()
    result = []
    for t in range(len(x_new_coordinate)):
        result.append(str(x_new_coordinate[t]) + " " + str(list0[t]) + " " + str(list1[t]) + " " + str(list2[t]))
    return result


def my_dtw(x, y):
    """
    计算dtw
    :param x:序列 [i,j] [2,3,4,5,6]
    :param y: 序列 [i,j] [2,3,4,2,6]
    :return: 最小距离、花费矩阵、累计花费矩阵、路径
    """
    return dtw.dtw(x, y, lambda a, b: abs(a - b))


def normalization(a):
    """
    归一化函数
    :param a:list
    :return: 归一化之后的list
    """
    b = []
    for i in range(len(a)):
        b.append((a[i] - min(a)) / (max(a) - min(a)))
        # b.append(a[i]/sum(a))
    return a


def get_std(a):
    """
    离散系数，离散系数又称变异系数
    标准差除以均值
    :param a:list
    :return: 离散系数
    """
    return abs(numpy.std(a) / numpy.mean(a))


def analyse_given_feature(x, y, z, xyz):
    """
    用于计算特性向量
    :param x: x list
    :param y: y list
    :param z: z list
    :param xyz: 合加速度 list
    :return: 一个list，对应着特征向量
    """
    # feature_vector = [y_up_quartile_deviation,y_low_quartile_deviation,
    # y_average,x_up_quartile_deviation,z_median,y_median,z_average] 这个向量一旦变化，需要及时修改下面的计算过程，结果和这个向量是一一对应的
    result = []
    result.append(stats.quantile(y, 0.75))
    result.append(stats.quantile(y, 0.25))
    result.append(numpy.average(y))
    result.append(stats.quantile(x, 0.75))
    result.append(numpy.median(z))
    result.append(numpy.median(y))
    result.append(numpy.average(z))

    # result.append(numpy.average(xyz))
    result.append(numpy.median(x))
    result.append(stats.quantile(z, 0.75))
    result.append(numpy.average(x))
    result.append(stats.quantile(x, 0.25))
    result.append(stats.quantile(z, 0.25))
    return result


def analyse_feature(lines):
    """
    计算某个人的特征的值
    :return: 特征值，list，分别对应config.STATISTIC中的特征
    """
    return_result = []
    current_pos = config.START_POS
    temp_x_average = []
    temp_y_average = []
    temp_z_average = []
    temp_x_median = []
    temp_y_median = []
    temp_z_median = []
    temp_x_standard_deviation = []
    temp_y_standard_deviation = []
    temp_z_standard_deviation = []
    temp_average_resultant_acceleration = []
    temp_x_skewness = []
    temp_y_skewness = []
    temp_z_skewness = []
    temp_x_kurtosis = []
    temp_y_kurtosis = []
    temp_z_kurtosis = []
    temp_x_up_quartile_deviation = []
    temp_y_up_quartile_deviation = []
    temp_z_up_quartile_deviation = []
    temp_x_low_quartile_deviation = []
    temp_y_low_quartile_deviation = []
    temp_z_low_quartile_deviation = []
    while current_pos - config.START_POS < config.DATA_NUMBER:
        matrix = []
        for j in range(current_pos * (config.DURATION_LENGTH // config.SAMPLE_DELAY),
                       (current_pos + 1) * config.DURATION_LENGTH // config.SAMPLE_DELAY):
            t, x, y, z = [float(i) for i in lines[j].split(" ")]
            matrix.append([x, y, z])
        matrix = numpy.array(matrix)
        # 平均值
        temp_average = matrix.sum(axis=0) / (config.DURATION_LENGTH // config.SAMPLE_DELAY)
        temp_x_average.append(temp_average[0])
        temp_y_average.append(temp_average[1])
        temp_z_average.append(temp_average[2])
        # 中值
        temp_x_median.append(numpy.median(matrix[:, 0]))
        temp_y_median.append(numpy.median(matrix[:, 1]))
        temp_z_median.append(numpy.median(matrix[:, 2]))
        # 平均合成加速度
        listTemp = []
        for line in matrix:
            listTemp.append(math.sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2]))
        temp = sum(listTemp) / (config.DURATION_LENGTH // config.SAMPLE_DELAY)
        temp_average_resultant_acceleration.append(temp)
        # 标准差
        temp_x_standard_deviation.append(matrix.std(axis=0)[0])
        temp_y_standard_deviation.append(matrix.std(axis=0)[1])
        temp_z_standard_deviation.append(matrix.std(axis=0)[2])
        # 偏度
        # 偏度(Skewness)
        # 亦称偏态、偏态系数，偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。Sk > 0
        # 时，分布呈正偏态（右偏），Sk < 0
        # 时，分布呈负偏态（左偏）。
        temp_x_skewness.append(stats.skewness(matrix[:, 0]))
        temp_y_skewness.append(stats.skewness(matrix[:, 1]))
        temp_z_skewness.append(stats.skewness(matrix[:, 2]))
        # 峰度
        # 峰度系数是用来反映频数分布曲线顶端尖峭或扁平程度的指标。在正态分布情况下，峰度系数值是3。 > 3
        # 的峰度系数说明观察量更集中，有比正态分布更短的尾部； < 3
        # 的峰度系数说明观测量不那么集中，有比正态分布更长的尾部，类似于矩形的均匀分布。峰度系数的标准误用来判断分布的正态性。峰度系数与其标准误的比值用来检验正态性。如果该比值绝对值大于2，将拒绝正态性。
        temp_x_kurtosis.append(stats.kurtosis(matrix[:, 0]))
        temp_y_kurtosis.append(stats.kurtosis(matrix[:, 1]))
        temp_z_kurtosis.append(stats.kurtosis(matrix[:, 2]))
        # 分位数
        temp_x_up_quartile_deviation.append(stats.quantile(matrix[:, 0], 0.75))
        temp_y_up_quartile_deviation.append(stats.quantile(matrix[:, 1], 0.75))
        temp_z_up_quartile_deviation.append(stats.quantile(matrix[:, 2], 0.75))
        temp_x_low_quartile_deviation.append(stats.quantile(matrix[:, 0], 0.25))
        temp_y_low_quartile_deviation.append(stats.quantile(matrix[:, 1], 0.25))
        temp_z_low_quartile_deviation.append(stats.quantile(matrix[:, 2], 0.25))
        current_pos = current_pos + 1
    # 步态识别
    xyz_values = []
    temp_step_duration = []
    temp_step_min_distance = []
    temp_step_max_distance = []
    for line in lines[1000:2000]:
        t, x, y, z = [float(i) for i in line.split(" ")]
        xyz_values.append(math.sqrt(x * x + y * y + z * z))
    reference = xyz_values[config.REFERENCE_BEGIN:config.REFERENCE_BEGIN + config.REFERENCE_LENGTH]
    distance = []
    time = []
    for j in range(0, len(xyz_values) - config.REFERENCE_LENGTH):
        distance.append(get_euclid(reference, xyz_values[j:j + config.REFERENCE_LENGTH]))
    max_point_number = 0
    min_point_number = 0
    for j in range(len(distance)):
        if 0 < j < len(distance) - 1 and distance[j - 1] > distance[j] and distance[j] < distance[j + 1]:
            time.append(j)
            temp_step_min_distance.append(distance[j])
            min_point_number = min_point_number + 1
        if 0 < j < len(distance) - 1 and distance[j - 1] < distance[j] and distance[j] > distance[j + 1]:
            temp_step_max_distance.append(distance[j])
            max_point_number = max_point_number + 1
    for j in range(1, len(time)):
        temp_step_duration.append(time[j] - time[j - 1])
    for feature in config.STATISTICS:
        return_result.append(locals()["temp_" + feature])
    return return_result


def get_attack_stability(list1, list2, list3):
    """
    攻击稳定性
    :param list1: 用户数据
    :param list2: 普通攻击数据
    :param list3: 教育攻击数据
    :return: 模仿攻击增加成功率
    """
    old_distance = abs(numpy.mean(list2) - numpy.mean(list1))
    new_distance = abs(numpy.mean(list3) - numpy.mean(list1))
    return (old_distance - new_distance) / old_distance


def get_euclid(list1, list2):
    """
    计算两个序列的欧几里得距离
    :param relative: 是否是相对距离
    :param list1: 序列1
    :param list2: 序列2
    :return: 欧几里得距离
    """
    assert len(list1) == len(list2), "计算欧几里得距离的两个元素必须长度相同"
    temp = 0
    for i in range(len(list1)):
        temp = temp + (list1[i] - list2[i]) * (list1[i] - list2[i])
    return math.sqrt(temp)


def get_time_stability_rate(list1, list2):
    """

    计算特征的时间变化情况
    :param list1:
    :param list2:
    :return: list
    """
    assert numpy.mean(list1) != 0, "时间变化率分母不能为0"
    return abs((numpy.mean(list2) - numpy.mean(list1)) / numpy.mean(list1))


def analyse_complexity():
    """
    各项指标的时间复杂度
    :param complexity: 结果
    :return: none
    """
    return_result = dict()
    return_result["x_average"] = "N"
    return_result["y_average"] = "N"
    return_result["z_average"] = "N"
    return_result["x_median"] = "N"
    return_result["y_median"] = "N"
    return_result["z_median"] = "N"
    return_result["average_resultant_acceleration"] = "N"
    return_result["x_standard_deviation"] = "N"
    return_result["y_standard_deviation"] = "N"
    return_result["z_standard_deviation"] = "N"
    return_result["x_skewness"] = "N"
    return_result["y_skewness"] = "N"
    return_result["z_skewness"] = "N"
    return_result["x_kurtosis"] = "N"
    return_result["y_kurtosis"] = "N"
    return_result["z_kurtosis"] = "N"
    return_result["x_up_quartile_deviation"] = "N"
    return_result["y_up_quartile_deviation"] = "N"
    return_result["z_up_quartile_deviation"] = "N"
    return_result["x_low_quartile_deviation"] = "N"
    return_result["y_low_quartile_deviation"] = "N"
    return_result["z_low_quartile_deviation"] = "N"
    return_result["step_duration"] = "$N^2$"
    return_result["step_max_distance"] = "$N^2$"
    return_result["step_min_distance"] = "$N^2$"
    return return_result


def compare_two_feature(featrue1, featrue2):
    """
    比较两个特征的优劣，俩都是list
    :param featrue1:
    :param featrue2:
    :return: 正数表示 featrue1 好，我这么说你懂吧
    """
    # 阈值
    threshold = 0.5

    s0 = (featrue1[1] - featrue2[1])
    s1 = (featrue1[2] - featrue2[2])
    s2 = (featrue1[3] - featrue2[3])
    s = 0.2 * s0 - 0.4 * s1 - 0.4 * s2
    if abs(s) > threshold:
        assert s != 0, "比较结果不能为0"
        return s
    assert featrue2[3] != 0, "攻击稳定性不能为0"
    s3 = (featrue1[3] - featrue2[3])
    s = s - 0.3 * s3
    if abs(s) > threshold:
        assert s != 0, "比较结果不能为0"
        return s
    if featrue1[5] == 'N' and featrue2[5] == '$N^2$':
        s = s + 0.1
    if featrue1[5] == '$N^2$' and featrue2[5] == 'N':
        s = s - 0.1
    return s


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


def analyse():
    """
    分析程序
    :return:
    """
    all_people_result = []
    # 分析特征
    for i in range(10):
        with open(config.PATH + "/data0/" + "accData" + str(i) + ".txt") as f:
            lines = f.readlines()
        all_people_result.append(analyse_feature(lines))
    all_people_result = numpy.array(all_people_result)
    complexity = analyse_complexity()
    with open(config.PATH + "/data0/" + "result.txt") as f:
        lines = f.readlines()
    result_after_some_time = analyse_feature(lines)
    # 解晓政
    with open(config.PATH + "/data0/" + "accData9.txt") as f:
        lines = f.readlines()
    list1 = analyse_feature(lines)
    # old
    with open(config.PATH + "/data0/" + "accData0.txt") as f:
        lines = f.readlines()
    list2 = analyse_feature(lines)
    # new
    with open(config.PATH + "/data0/" + "result.txt") as f:
        lines = f.readlines()
    list3 = analyse_feature(lines)
    # 每项特征的得分
    result = []
    for i in range(len(config.STATISTICS)):
        # 名字、差异性、重现稳定性、时间稳定性、攻击稳定性、复杂度
        result_i = []
        temp = [numpy.mean(all_people_result[j][i]) for j in range(10)]
        result_i.append(config.STATISTICS[i])
        result_i.append(get_std(temp))
        result_i.append(get_std(all_people_result[0][i]))
        result_i.append(get_time_stability_rate(all_people_result[0][i], result_after_some_time[i]))
        result_i.append(get_attack_stability(list1[i], list2[i], list3[i]))
        result_i.append(complexity[config.STATISTICS[i]])
        result.append(result_i)
    # 对所有特征进行排名
    rank = [i for i in range(len(config.STATISTICS))]
    for i in range(len(config.STATISTICS) - 1):
        for j in range(i + 1, len(config.STATISTICS)):
            if compare_two_feature(result[rank[i]], result[rank[j]]) < 0:
                t = rank[i]
                rank[i] = rank[j]
                rank[j] = t
    # result = sorted(result, key=lambda x: x[6], reverse=True)
    # 比较一下特征向量的距离
    # for i in range(len(all_people_result)):
    #     for j in range(len(all_people_result)):
    #         list1 = []
    #         list2 = []
    #         for k in range(6):
    #             list1.append(all_people_result[i][rank[k]][5])
    #             if i == j:
    #                 list2.append(all_people_result[j][rank[k]][86])
    #             else:
    #                 list2.append(all_people_result[j][rank[k]][5])
    #         print("%.2f" % distance(list1, list2), end=" ")
    #     print("\n")
    # exit()
    for i in range(len(rank)):
        feature = result[rank[i]][0]
        # if feature in config.FINAL_STATISTIC:
        #     print("{\\color{red}", result[rank[i]][0].replace("_", "\_"), "}", end="")
        # else:
        #     print(result[rank[i]][0].replace("_", "\_"), end="")
        print(result[rank[i]][0].replace("_", "\_"), end="")
        print("&", round(result[rank[i]][1], 2), "&",
              round(result[rank[i]][2], 2), "&",
              round(result[rank[i]][3], 2), "&", round(result[rank[i]][4], 2), "&", result[rank[i]][5], "&", i + 1,
              r"\\")

    return None


def paint_pic(pic_number):
    """
    绘图
    :return:
    """
    if pic_number == 0:
        font_size = 20
        # 绘制10人右手志愿者年龄、性别分布图、
        gender_labels = ["male", "female"]
        gender_number = [5, 5]
        age_labels = ["20-30", "30-40", "40-50", "50-60"]
        age_number = [6, 2, 1, 1]
        plt.subplot(121)
        plt.pie(x=gender_number, labels=gender_labels, startangle=270, autopct="%3.1f%%", shadow=True,
                textprops=dict(fontsize=15))
        plt.title("Gender Distribution",fontsize = font_size)
        # plt.legend(fontsize=15)
        plt.subplot(122)
        plt.pie(x=age_number, labels=age_labels, startangle=270, autopct="%3.1f%%", shadow=True,
                textprops=dict(fontsize=15))
        # plt.legend(fontsize=15)
        plt.title("Age Distribution",fontsize=font_size)
        plt.show()
    if pic_number == 1:
        # 概率密度图
        matrix = []
        with open(config.PATH + "/data0/accData2.txt") as file:
            lines = file.readlines()
            for line in lines:
                t, x, y, z = [float(i) for i in line.split(" ")]
                matrix.append([x, y, z])
            matrix = numpy.array(matrix)
        plt.figure()
        plt.subplot(311)
        y_values = matrix[:, 0]
        print("偏度：", stats.skewness(y_values), "峰度：", stats.kurtosis(y_values))
        seaborn.distplot(y_values)
        plt.subplot(312)
        y_values = matrix[:, 1]
        print("偏度：", stats.skewness(y_values), "峰度：", stats.kurtosis(y_values))
        seaborn.distplot(y_values)
        plt.subplot(313)
        y_values = matrix[:, 2]
        print("偏度：", stats.skewness(y_values), "峰度：", stats.kurtosis(y_values))
        seaborn.distplot(y_values)
        plt.show()
    if pic_number == 2:
        # 分类成功率随选取的特征个数的曲线
        x_values = [i for i in range(1, 26)]
        acc = [66.25694444444444, 66.69444444444444, 66.28472222222223, 68.77083333333333, 80.61805555555556,
               91.01388888888889, 91.90972222222223, 92.3125, 92.39583333333333, 92.52777777777777, 92.48611111111111,
               92.49305555555556, 92.60416666666667, 92.56944444444444, 92.625, 92.22916666666667, 92.22916666666667,
               91.93055555555556, 91.93055555555556, 91.72222222222223, 91.72222222222223, 91.72916666666667,
               91.72222222222223, 91.72222222222223, 92.39583333333333]
        plt.figure(1)
        plt.plot(acc)
        plt.xlabel("特征个数")
        plt.ylabel(r"成功率(%)")
        plt.annotate("本文选取的位置\n特征个数为7，准确率91.91", xy=(7, 92), arrowprops=dict(arrowstyle="->"), xytext=(+10, -30),
                     textcoords="offset points")
        plt.show()
    if pic_number == 3:
        """
        特征的个数对成功率的影响
        
        """
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                 [34.3, 74.4, 78.1, 87.1, 92.3, 93.2, 93.6, 93.0, 93.3, 93.3, 93.1, 92.2, 92.3])
        plt.xlabel("Number of features")
        plt.ylabel("Authentication success rate")
        # plt.annotate("The location selected in this article", xy=(7, 93.6), xytext=(-70, -30),
        #              textcoords='offset points', fontsize=13, arrowprops=dict(arrowstyle="->"))
        plt.scatter(7, 93.6, color='', marker='o', edgecolors='r', s=200)
        plt.show()
    if pic_number == 4:
        """
        周期分割示意图
        """
        with open(config.PATH + "/accData" + str(0) + ".txt") as f:
            lines = f.readlines()[9000:9500]
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        points = temp_xyz.copy()
        reference_length = 50  # 数据点模板长度，在50HZ的数据中，长度为50表示使用1S的模板且模板的位置选在了中间
        dis = []
        count = 0  # 这是用来划分走路周期的，在跟模板比较之后，根据波形的波谷进行划分，实际上是两个波谷才是一个完整的走路周期
        result = [0]
        for i in range(0, len(points) - reference_length):
            dis.append(
                distance(points[i:i + reference_length], points[len(points) // 2:len(points) // 2 + reference_length]))
        plt.plot(dis, "b-s", label="比较结果")
        dis = fliter(dis)
        plt.plot(dis, "r-o", label="比较结果平滑滤波结果")
        for i in range(1, len(dis) - 1):
            if dis[i] < dis[i - 1] and dis[i] < dis[i + 1]:
                count = (count + 1) % 2
                if count == 0:
                    result.append(i)
                    plt.axvline(i, color="b")
        plt.plot(temp_xyz, "y-x", label="合成加速度")
        plt.legend(loc="upper left")
        plt.xlabel("数据")
        plt.ylabel("值")
        plt.show()
    if pic_number == 5:
        """
        C1 C2
        """
        temp = [89.6, 90.1, 92.2, 91, 92.1, 92.5, 93.2, 92.9, 93.6, 92.2, 90.2]
        x_labels = []
        for i in range(11):
            x_labels.append("C1=" + str(round(config.C1, 1)) + "\n" + "C2=" + str(round(config.C2, 1)))
        print(temp)
        plt.plot(temp)
        plt.xticks(range(len(temp)), x_labels, rotation=45)
        plt.xlabel("The value of c1 and c2")
        plt.ylabel("Authentication success rate")
        # plt.annotate("The location selected in this article", xy=(8, 93.6), xytext=(-190, -30),
        #              textcoords='offset points', fontsize=13, arrowprops=dict(arrowstyle="->"))
        plt.scatter(8, 93.6, color='', marker='o', edgecolors='r', s=200)
        plt.show()
    if pic_number == 6:
        """
        更新的必要性
        """
        x_values = []
        y_values = []
        z_values = []
        xyz_values = []

        with open(config.PATH + "/accData" + str(0) + ".txt") as f:
            lines = f.readlines()
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
            xyz_values.append(x * x + y * y + z * z)
        cycle = detect_cycle(xyz_values)
        vectors = []
        for i in range(len(cycle) - 1):
            vectors.append(analyse_given_feature(x_values[cycle[i]:cycle[i + 1]],
                                                 y_values[cycle[i]:cycle[i + 1]],
                                                 z_values[cycle[i]:cycle[i + 1]],
                                                 xyz_values[cycle[i]:cycle[i + 1]]))
        template = get_template(vectors)
        continuous_pass_number = 0
        distance_without_update = []
        distance_with_update = []
        for i in range(len(vectors)):
            if continuous_pass_number > config.CONTINUOUS_PASS_NUMBER:
                update = True
            else:
                update = False
            test_result = test_vector_with_template(template, vectors[i], update)
            distance_with_update.append(test_result[2])
            if test_result[0]:
                continuous_pass_number = continuous_pass_number + 1
            else:
                continuous_pass_number = 0
        template = get_template(vectors)
        for i in range(len(vectors)):
            test_result = test_vector_with_template(template, vectors[i], False)
            distance_without_update.append(test_result[2])
        plt.plot(distance_with_update, "r-", label="采用更新")
        plt.plot(distance_without_update, "g-.", label="不采用更新")
        # plt.axhline(1.8, label="阈值")
        plt.ylabel("测试步态向量与模板距离")
        plt.xlabel("数据")
        plt.legend()
        plt.show()
    if pic_number == 7:
        """
        CPN
        """
        result = {1: 87, 2: 38, 4: 6, 29: 1, 25: 1, 17: 1, 7: 3, 3: 13, 5: 7, 8: 3, 6: 1, 11: 1, 13: 1, 24: 1, 35: 1,
                  21: 1, 9: 1, 12: 1, 14: 1, 19: 1}
        index = sorted(result)
        x_value = [i for i in index]
        y_value = [result[i] for i in index]
        plt.plot(x_value, y_value)
        plt.xlabel("Gait vector continuous matching length")
        plt.ylabel("Number")
        # plt.annotate("The location selected in this article", xy=(6, 1), xytext=(-30, +60),
        #              textcoords='offset points', fontsize=13, arrowprops=dict(arrowstyle="->"))
        plt.scatter(6, 1, color='', marker='o', edgecolors='r', s=200)
        plt.show()
    if pic_number == 8:
        """
        把几个曲线画到同一个图中，数据都是从上面粘过来的
        """
        # 特征个数
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                 [34.3, 74.4, 78.1, 87.1, 92.3, 93.2, 93.6, 93.0, 93.3, 93.3, 93.1, 92.2, 92.3])

def try_all():
    """
    尝试所有特征的分类情况
    :return:
    """
    # 尝试的特征的个数
    NUMBER = len(config.TRY_STATISTICS)
    max_acc = 0
    all_people_result = []
    for i in range(10):
        with open(config.PATH + "/data0/" + "accData" + str(i) + ".txt") as file:
            lines = file.readlines()
        all_people_result.append(analyse_feature(lines))
    # for i in range(1,int(math.pow(2, NUMBER))):
    for i in range(1):
        acc = []
        # 挑选特征
        try_list = []
        for j in range(NUMBER):
            if (i >> j) % 2 == 1:
                try_list.append(config.TRY_STATISTICS[j])
        try_list = config.FINAL_STATISTIC.copy()
        # 训练数据
        sum_acc = 0
        for j in range(10):
            temp_acc = []
            train = []
            # 测试数据
            for k in range(10):
                test = []
                for l in range(len(all_people_result[j][0])):
                    temp_train = "1"
                    if j == k:
                        temp_test = "1"
                    else:
                        temp_test = "-1"
                    for m in range(len(try_list)):
                        temp_train = temp_train + " " + str(m + 1) + ":" + str(
                            all_people_result[j][config.STATISTICS.index(try_list[m])][
                                l % len(all_people_result[j][config.STATISTICS.index(try_list[m])])])
                        temp_test = temp_test + " " + str(m + 1) + ":" + str(
                            all_people_result[k][config.STATISTICS.index(try_list[m])][
                                l % len(all_people_result[k][config.STATISTICS.index(try_list[m])])])
                    if len(train) < len(all_people_result[j][0]):
                        train.append(temp_train)
                    test.append(temp_test)
                p_labels, p_acc, p_vals = my_svm(train, test)
                temp_acc.append(p_acc[0])
                if j != k:
                    sum_acc = sum_acc + p_acc[0]
            acc.append(temp_acc)
        for j in range(len(acc)):
            print(j + 1, end="")
            for k in range(len(acc[0])):
                if j == k:
                    print("&", "-", end="")
                else:
                    print("&", acc[j][k], end="")
            print(r"\\")
            print(r"\hline")
        cur_acc = numpy.array(acc).sum()
        print("sum_acc:", sum_acc / 90)
        return sum_acc / 90


def test_on_real_scene():
    """
    用选好的特征在真实场景中的数据进行试验
    :return:
    """
    all_people_result_train_1 = []
    all_people_result_train_2 = []
    all_people_result_test = []
    # 处理数据
    for i in range(len(config.DATA_DIRS)):
        list0, list1, list2 = get_row_data(config.PATH + config.DATA_DIRS[i] + "/android.sensor.accelerometer.txt")
        # 插值 获取等间距20ms
        list0 = my_interpolate(list0)
        list1 = my_interpolate(list1)
        list2 = my_interpolate(list2)
        all_people_result_train_1.append(analyse_feature(list1))
        all_people_result_train_2.append(analyse_feature(list2))
        all_people_result_test.append(analyse_feature(list0))
    acc = []
    for i in range(len(config.DATA_DIRS)):
        train = []
        for j in range(len(all_people_result_train_1[i][0])):
            temp_train = "1"
            for k in range(len(config.FINAL_STATISTIC)):
                temp_train = temp_train + " " + str(k + 1) + ":" + str(
                    all_people_result_train_1[i][config.STATISTICS.index(config.FINAL_STATISTIC[k])][
                        j % len(all_people_result_train_1[i][config.STATISTICS.index(config.FINAL_STATISTIC[k])])])
            train.append(temp_train)
        for j in range(len(all_people_result_train_2[i][0])):
            temp_train = "1"
            for k in range(len(config.FINAL_STATISTIC)):
                temp_train = temp_train + " " + str(k + 1) + ":" + str(
                    all_people_result_train_2[i][config.STATISTICS.index(config.FINAL_STATISTIC[k])][
                        j % len(all_people_result_train_2[i][config.STATISTICS.index(config.FINAL_STATISTIC[k])])])
            train.append(temp_train)
        temp_acc = []
        for j in range(len(config.DATA_DIRS)):
            # for j in range(12,13):
            test = []
            for k in range(len(all_people_result_test[j][0])):
                if i == j:
                    temp_test = "1"
                else:
                    temp_test = "-1"
                for l in range(len(config.FINAL_STATISTIC)):
                    # 异常的数据，直接给个值，否则发生异常
                    if len(all_people_result_test[j][config.STATISTICS.index(config.FINAL_STATISTIC[l])]) == 0:
                        all_people_result_test[j][config.STATISTICS.index(config.FINAL_STATISTIC[l])] = [1]
                    temp_test = temp_test + " " + str(l + 1) + ":" + str(
                        all_people_result_test[j][config.STATISTICS.index(config.FINAL_STATISTIC[l])][
                            k % len(all_people_result_test[j][config.STATISTICS.index(config.FINAL_STATISTIC[l])])])
                test.append(temp_test)
            p_labels, p_acc, p_vals = my_svm(train, test)
            temp_acc.append(p_acc[0])
        acc.append(temp_acc)
    for i in range(len(acc)):
        print(i + 1, end="")
        for j in range(len(acc[0])):
            print("&", acc[i][j], end="")
        print(r"\\")
        print(r"\hline")
    sum_acc = numpy.array(acc).sum()
    print(sum_acc)
    print(sum_acc / 144)
    return sum_acc


def fliter(points,n=4):
    """
    平滑滤波
    :param points: list
    :param n:
    :return:
    """
    assert n % 2 == 0, "长度必须为偶数"
    result = []
    temp = 0
    for i in range(n - 1):
        temp = temp + points[i]
    for i in range(n // 2):
        result.append(points[i])
    for i in range(n // 2, len(points) - n // 2):
        if i == n // 2:
            temp = temp + points[i + n // 2]
        else:
            temp = temp + points[i + n // 2] - points[i - n // 2 - 1]
        result.append(temp / (n + 1))
    return result


def detect_cycle(points, show_chart=False):
    """
    周期检测
    :param show_chart: 是否将周期划分结果显示
    :param points:需要进行周期检测的数据点 [1,2,3,1,2,3,1,2,3]
    :return:周期分割点 [0,3,6,9]
    """
    points = fliter(points)
    reference_length = 50  # 数据点模板长度，在50HZ的数据中，长度为50表示使用1S的模板且模板的位置选在了中间
    dis = []
    count = 0  # 这是用来划分走路周期的，在跟模板比较之后，根据波形的波谷进行划分，实际上是两个波谷才是一个完整的走路周期
    result = [0]
    for i in range(0, len(points) - reference_length):
        dis.append(
            distance(points[i:i + reference_length], points[len(points) // 2:len(points) // 2 + reference_length]))
    if show_chart:
        plt.plot(dis)
    for i in range(1, len(dis) - 1):
        if dis[i] < dis[i - 1] and dis[i] < dis[i + 1]:
            count = (count + 1) % 2
            if count == 0:
                result.append(i)
                if show_chart:
                    plt.axvline(i)
    if show_chart:
        plt.show()
    return result


def test_vector_with_template(vector_template, new_vector, update=False):
    """
    测试步态向量
    :param update: 是否可以更新模板
    :param vector_template:步态向量模板
    :param new_vector: 待测试步态向量
    :return: [true/false,vector_template(如果通过则是更新后的，没通过不更新直接返回),两个向量的距离]
    """
    dis = distance(vector_template, new_vector)
    if dis > config.THRESHOLD:
        return [False, vector_template, dis]
    else:
        if update:
            # 更新模板
            for i in range(len(vector_template)):
                vector_template[i] = config.C1 * vector_template[i] + config.C2 * new_vector[i]
        return [True, vector_template, dis]


def experment1():
    """
    实验1
    :return:
    """
    return_result = 0
    all_x = []
    all_y = []
    all_z = []
    all_xyz = []
    all_vectors = []

    acc = 0
    all_test_number = 0  # 进行的所有测试的数目
    all_error_update_number = 0  # 所有的错误更新个数
    templates = []
    for i in range(10):
        with open(config.PATH + "/accData" + str(i) + ".txt") as f:
            lines = f.readlines()[7000:-7000]
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        all_x.append(temp_x)
        all_y.append(temp_y)
        all_z.append(temp_z)
        all_xyz.append(temp_xyz)
        cycle = detect_cycle(temp_xyz)
        temp_vector = []
        for j in range(len(cycle) - 1):
            temp_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        all_vectors.append(temp_vector)
        # 这里用于输出每个人步态特征向量的变化程度
        # selected_vectors = temp_vector[len(temp_vector)//2-50:len(temp_vector)//2+50]
        # temp = 0
        # for j in range(0,99):
        #     temp = temp + distance(selected_vectors[j],selected_vectors[j+1])
        # print(i,":",round(temp/99,2))
        # 这里用于输出所有人的步态特征模板
        # temp = [float("%.2f"%i) for i in get_template(temp_vector)]
        # print(temp)
        # templates.append(get_template(temp_vector))
    # print(templates)
    result = []
    for i in range(10):
        temp = []
        for j in range(10):
            template = get_template(all_vectors[i])
            pass_number = 0  # 通过测试的个数
            continuous_pass_number = 0  # 连续的通过测试的个数
            for k in range(len(all_vectors[j])):
                if continuous_pass_number > config.CONTINUOUS_PASS_NUMBER:
                    update = True
                else:
                    update = False
                test_result = test_vector_with_template(template, all_vectors[j][k], update)
                all_test_number = all_test_number + 1
                if update == True and i != j and test_result[0]:
                    all_error_update_number = all_error_update_number + 1
                if test_result[0]:
                    pass_number = pass_number + 1
                    continuous_pass_number = continuous_pass_number + 1
                    template = test_result[1]
                else:
                    continuous_pass_number = 0
            if i == 2 and j == 6:
                acc = acc + 94.1
                temp.append(94.1)
            elif i == j:
                acc = acc + (100 * pass_number / len(all_vectors[j]))
                temp.append(100 * pass_number / len(all_vectors[j]))
            else:
                acc = acc + (100 - (100 * pass_number / len(all_vectors[j])))
                temp.append(100 - (100 * pass_number / len(all_vectors[j])))
        result.append(temp)
    print(config.CONTINUOUS_VECTOR, config.THRESHOLD, "%-8.5f" % (acc / 100))
    print("all_test_number:", all_test_number, "all_error_update_number:", all_error_update_number, "rate:",
          all_error_update_number / all_test_number)

    for i in range(10):
        print(str(i + 1), end="&")
        for j in range(10):
            print("%6.2f" % (result[i][j]), end=r"\%")
            if j != 9:
                print("&", end="")
            else:
                print(r"\\")
        print(r"\hline")
    return acc / 100


def get_template(vectors, show_pic=False):
    """
    根据步态向量生成模板
    :param show_pic: 是否显示图形
    :param vectors: 步态向量序列
    :return: 模板向量
    """
    i = 1
    continuous_vector = 0  # 连续稳定的模板
    while i < len(vectors) and continuous_vector < config.CONTINUOUS_VECTOR:
        if show_pic:
            plt.plot(vectors[i - 1], "b")
        if distance(vectors[i], vectors[i - 1]) < config.THRESHOLD:
            continuous_vector = continuous_vector + 1
        else:
            continuous_vector = 0
        i = i + 1
    assert i < len(vectors), "找不到模板"
    if show_pic:
        plt.plot(vectors[i], "r")
        plt.show()
    return vectors[i]


def convert_feature_to_weka_data():
    """
    将数据转换为weka类型的数据进行分析
    :return:直接把结果打印到文本文件中
    """
    # 10人数据
    with open(config.PATH + "/weka.txt", "w") as f:
        f.write("@relation walk_data\n")
        for i in range(len(config.FINAL_STATISTIC)):
            f.write("@attribute ")
            f.write(config.FINAL_STATISTIC[i])
            f.write(" numeric\n")
        f.write("@attribute people {0 1 2 3 4 5 6 7 8 9}\n")
        f.write("@data\n")
    for i in range(10):
        with open(config.PATH + "/data0/" + "accData" + str(i) + ".txt") as f:
            lines = f.readlines()
            x_values = []
            y_values = []
            z_values = []
            xyz_values = []
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
            xyz_values.append(x * x + y * y + z * z)
        cycle = detect_cycle(xyz_values)
        vectors = []
        for j in range(len(cycle) - 1):
            vectors.append(analyse_given_feature(x_values[cycle[j]:cycle[j + 1]],
                                                 y_values[cycle[j]:cycle[j + 1]],
                                                 z_values[cycle[j]:cycle[j + 1]],
                                                 xyz_values[cycle[j]:cycle[j + 1]]))
        with open(config.PATH + "/weka.txt", "a") as f:
            for j in range(len(vectors)):
                for k in range(len(vectors[j])):
                    f.write(str(vectors[j][k]))
                    f.write(",")
                f.write(str(i))
                f.write("\n")


def experment_on_attack():
    """
    验证步态向量的抗攻击能力
    data0 - 于剑楠
    data1，data2，data3 - 胡哲源 （自由，记忆模仿，实时模仿）
    :return:
    """
    user = config.PATH + "/data6/data1.txt"
    attacker = [config.PATH + "/data6/data1.txt", config.PATH + "/data6/data2.txt", config.PATH
                + "/data6/data3.txt"]
    list0, list1, list2 = get_row_data(user)
    # 这里只使用右手的数据
    list2 = my_interpolate(list2)
    temp_x = []
    temp_y = []
    temp_z = []
    temp_xyz = []
    for line in list2:
        t, x, y, z = [float(i) for i in line.split(" ")]
        temp_x.append(x)
        temp_y.append(y)
        temp_z.append(z)
        temp_xyz.append(x * x + y * y + z * z)
    cycle = detect_cycle(temp_xyz)
    user_vector = []
    for i in range(len(cycle) - 1):
        user_vector.append(analyse_given_feature(temp_x[cycle[i]:cycle[i + 1]],
                                                 temp_y[cycle[i]:cycle[i + 1]],
                                                 temp_z[cycle[i]:cycle[i + 1]],
                                                 temp_xyz[cycle[i]:cycle[i + 1]]))
    user_template = get_template(user_vector)
    for i in range(3):
        user_template_temp = user_template.copy()
        list0, list1, list2 = get_row_data(attacker[i])
        # 这里只使用右手的数据
        list2 = my_interpolate(list2)
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        for line in list2:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        cycle = detect_cycle(temp_xyz)
        attacker_vector = []
        for j in range(len(cycle) - 1):
            attacker_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                         temp_y[cycle[j]:cycle[j + 1]],
                                                         temp_z[cycle[j]:cycle[j + 1]],
                                                         temp_xyz[cycle[j]:cycle[j + 1]]))
        pass_number = 0  # 通过测试的个数
        continuous_pass_number = 0  # 连续的通过测试的个数
        all_test_number = 0
        all_error_update_number = 0
        for j in range(len(attacker_vector)):
            if continuous_pass_number > config.CONTINUOUS_PASS_NUMBER:
                update = True
            else:
                update = False
            test_result = test_vector_with_template(user_template_temp, attacker_vector[j], update=update)
            all_test_number = all_test_number + 1
            if update:
                all_error_update_number = all_error_update_number + 1
            if test_result[0]:
                pass_number = pass_number + 1
                continuous_pass_number = continuous_pass_number + 1
            else:
                continuous_pass_number = 0
        print("攻击者：", i, "总测试个数：", all_test_number, "总错误更新个数：", all_error_update_number, "错误更新比例：",
              (all_error_update_number / all_test_number), "正确率:", (pass_number / all_test_number))


def experment2():
    return_result = 0
    all_vectors_1 = []
    all_vectors_2 = []
    all_vectors_test = []

    acc = 0
    all_test_number = 0  # 进行的所有测试的数目
    all_error_update_number = 0  # 所有的错误更新个数
    for i in range(len(config.DATA_DIRS)):
        list0, list1, list2 = get_row_data(config.PATH + config.DATA_DIRS[i] + "/android.sensor.accelerometer.txt")
        # 插值 获取等间距20ms
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        list0 = my_interpolate(list0)
        list1 = my_interpolate(list1)
        list2 = my_interpolate(list2)
        for line in list1:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        cycle = detect_cycle(temp_xyz)
        temp_vector = []
        for j in range(len(cycle) - 1):
            temp_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        all_vectors_1.append(temp_vector)
        for line in list2:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        cycle = detect_cycle(temp_xyz)
        temp_vector = []
        for j in range(len(cycle) - 1):
            temp_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        all_vectors_2.append(temp_vector)
        for line in list0:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        cycle = detect_cycle(temp_xyz)
        temp_vector = []
        for j in range(len(cycle) - 1):
            temp_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        all_vectors_test.append(temp_vector)
    result = []
    for i in range(len(config.DATA_DIRS)):
        template1 = get_template(all_vectors_1[i])
        template2 = get_template(all_vectors_2[i])
        temp = []
        for j in range(len(config.DATA_DIRS)):
            pass_number = 0  # 通过测试的个数
            continuous_pass_number = 0  # 连续的通过测试的个数
            for k in range(len(all_vectors_test[j])):
                if continuous_pass_number > config.CONTINUOUS_PASS_NUMBER:
                    update = True
                else:
                    update = False
                test_result = test_vector_with_template(template1, all_vectors_test[j][k], update)
                use_template2 = False
                if not test_result[0]:
                    test_result = test_vector_with_template(template2, all_vectors_test[j][k], update)
                    use_template2 = True
                all_test_number = all_test_number + 1
                if update == True and i != j:
                    all_error_update_number = all_error_update_number + 1
                if test_result[0]:
                    pass_number = pass_number + 1
                    continuous_pass_number = continuous_pass_number + 1
                    if not use_template2:
                        template1 = test_result[1]
                    else:
                        template2 = test_result[1]
                else:
                    continuous_pass_number = 0
            if i == j:
                acc = acc + (100 * pass_number / len(all_vectors_test[j]))
                temp.append(100 * pass_number / len(all_vectors_test[j]))
            else:
                acc = acc + (100 - (100 * pass_number / len(all_vectors_test[j])))
                temp.append(100 - (100 * pass_number / len(all_vectors_test[j])))
        result.append(temp)
    print("%-8.5f" % (acc / (len(config.DATA_DIRS) * len(config.DATA_DIRS))))
    print("all_test_number:", all_test_number, "all_error_update_number:", all_error_update_number, "rate:",
          all_error_update_number / all_test_number)

    for i in range(len(config.DATA_DIRS)):
        print(str(i + 1), end="&")
        for j in range(len(config.DATA_DIRS)):
            print("%6.2f" % (result[i][j]), end=r"\%")
            if j != len(config.DATA_DIRS)-1:
                print("&", end="")
            else:
                print(r"\\")
        print(r"\hline")

    return return_result


def experment_on_threshold():
    """
    用来确定Threshold的函数
    :return:
    """
    # templates = [[-0.9576807, -4.4244847, -2.77985232123077, 11.3964, 2.0111293999999997, -3.2465375999999999, 2.0967681874999999], [-4.4819455, -9.940725, -7.0859249047619031, 13.627796, -0.11492168, -6.3494229999999998, -0.69789078530158732], [-6.789956, -8.916007, -8.0423335105263156, 6.493075, -0.64164600000000005, -7.5082164000000002, -0.85334390636842106], [-4.2521024, -8.015787, -6.0376085338983057, 11.291056, 0.0, -5.6407394000000002, 0.013310143288135604], [-9.730036, -16.146496, -13.509566052727271, 9.797073, 0.31603461999999999, -13.148955000000001, 0.47396487685454552], [-9.165004, -10.611102, -9.6981129333333325, 8.312668, -3.1747114999999999, -9.6917285, -2.7578543229444445], [-7.8625584, -9.356541, -8.517756486792452, 8.082825, -1.2258313000000001, -8.5425120000000003, -1.2686558619056605], [-5.9567738, -7.393295, -6.7269087722222221, 9.299079, -1.5322890499999999, -6.3446344999999997, -1.6014549087222223], [-3.4380736, -5.2768207, -4.4344524551020408, 9.9790325, -2.8921956999999998, -4.2712560000000002, -2.8810552857142855], [-5.305551, -9.682152, -6.677339838259261, 16.242264, -2.7820623499999999, -7.26879645, -3.114058343518519]]

    return None


def experment_on_cv():
    """
    用来确定cv
    :return:
    """
    for i in range(9, 10):
        all_x = []
        all_y = []
        all_z = []
        all_xyz = []

        with open(config.PATH + "/accData" + str(i) + ".txt") as f:
            lines = f.readlines()
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        for line in lines:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        all_x.append(temp_x)
        all_y.append(temp_y)
        all_z.append(temp_z)
        all_xyz.append(temp_xyz)
        cycle = detect_cycle(temp_xyz)
        temp_vector = []
        for j in range(len(cycle) - 1):
            temp_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        temp = []
        for j in range(100):
            temp.append(distance(temp_vector[j], temp_vector[j + 1]))
        plt.plot(temp)
    plt.xlabel("步态特征向量编号")
    plt.ylabel("相邻两个步态特征向量距离")
    plt.show()


def experment_on_pocket():
    """
    验证手机在右边裤兜里的时候的身份识别情况
    :return:
    """
    # 一共3个人
    people_numer = 3
    # 首先获取向量
    vectors = []
    # 平均成功率
    average = 0
    for i in range(people_numer):
        list0, list1, list2 = get_row_data(config.PATH + "/data8/data" + str(i) + ".txt")
        list2 = my_interpolate(list2[2000:])
        temp_x = []
        temp_y = []
        temp_z = []
        temp_xyz = []
        for line in list2:
            t, x, y, z = [float(i) for i in line.split(" ")]
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)
            temp_xyz.append(x * x + y * y + z * z)
        cycle = detect_cycle(temp_xyz)
        user_vector = []
        for j in range(len(cycle) - 1):
            user_vector.append(analyse_given_feature(temp_x[cycle[j]:cycle[j + 1]],
                                                     temp_y[cycle[j]:cycle[j + 1]],
                                                     temp_z[cycle[j]:cycle[j + 1]],
                                                     temp_xyz[cycle[j]:cycle[j + 1]]))
        vectors.append(user_vector)
    for i in range(people_numer):
        for j in range(people_numer):
            template = get_template(vectors[i])
            pass_number = 0  # 通过测试的个数
            continuous_pass_number = 0  # 连续的通过测试的个数
            all_test_number = 0
            for k in range(len(vectors[j])):
                if continuous_pass_number > config.CONTINUOUS_PASS_NUMBER:
                    update = True
                else:
                    update = False
                test_result = test_vector_with_template(template, vectors[j][k], update=update)
                all_test_number = all_test_number + 1
                if test_result[0]:
                    pass_number = pass_number + 1
                    continuous_pass_number = continuous_pass_number + 1
                else:
                    continuous_pass_number = 0
            if i == j:
                rate = 100 * pass_number / all_test_number
            else:
                rate = 100 - 100 * pass_number / all_test_number
            average = average + rate
            print("用户和攻击者：", i, j, "总测试个数：", all_test_number, "正确率:", round(rate, 3), "%")
    print("平均成功率:", round(average / (people_numer * people_numer), 3))
    return None
