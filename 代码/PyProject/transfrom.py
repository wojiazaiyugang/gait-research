import numpy
import math
import os
import random
from config import DURATION,SAMPLE_DELAY,PATH,STATISTICS,LOC
#使用训练数据的数量百分比
PERCENT = 1

#数据转换
def transform(fromFile,toFile,label):
    global LOC
    resultFile = open(PATH+toFile, "w")
    f = open(PATH+fromFile, "r")
    lines = f.readlines()
    f.close()
    while (len(lines) >= LOC + DURATION // SAMPLE_DELAY):
        matrixList = []
        for j in range(LOC, LOC + DURATION // SAMPLE_DELAY):
            t, x, y, z,*l = lines[j][0:len(lines[j])].split(" ")
            listTemp = [float(x), float(y), float(z)]
            matrixList.append(listTemp)
        matrix = numpy.array(matrixList)
        # 均值
        xAverage, yAverage, zAverage = matrix.sum(axis=0) / (DURATION // SAMPLE_DELAY)
        # 标准差
        xStandardDeviation, yStandardDeviation, zStandardDeviation = matrix.std(axis=0)
        # 平均绝对误差
        xAverageAbsoluteDifference = ((matrix[:, 0] - xAverage).sum()) / (DURATION // SAMPLE_DELAY)
        yAverageAbsoluteDifference = ((matrix[:, 1] - yAverage).sum()) / (DURATION // SAMPLE_DELAY)
        zAverageAbsoluteDifference = ((matrix[:, 2] - zAverage).sum()) / (DURATION // SAMPLE_DELAY)
        listTemp = []
        for line in matrixList:
            listTemp.append(math.sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2]))
        # 平均合成加速度
        averageResultantAcceleration = sum(listTemp) / (DURATION // SAMPLE_DELAY)
        #最大合成加速度
        maxResultantAcceleration = max(listTemp)
        #最小合成加速度
        minResultantAcceleration = min(listTemp)
        # 最值
        xMax, yMax, zMax, = numpy.max(matrix, axis=0)
        xMin, yMin, zMin = numpy.min(matrix, axis=0)
        xMaxDifference = xMax - xMin
        yMaxDifference = yMax - yMin
        zMaxDifference = zMax - zMin
        # 绝对值最值
        xAbsoluteMax, yAbsoluteMax, zAbsoluteMax = numpy.max(numpy.fabs(matrix), axis=0)
        xAbsoluteMin, yAbsoluteMmin, zAbsoluteMmin = numpy.min(numpy.fabs(matrix), axis=0)
        resultFile.write(str(int(label)))
        for i in range(len(STATISTICS)):
            resultFile.write(" " + str(i + 1) + ":" + str(locals()[STATISTICS[i]]))
        resultFile.write("\n")
        LOC = LOC + DURATION // SAMPLE_DELAY
    resultFile.flush()
    resultFile.close()