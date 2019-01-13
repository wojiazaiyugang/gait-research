"""
主函数，跑一个网络需要三句话
network = WalkNetwork() # 新建一个网络，目前支持MnistNetwork和WalkNetwork
network.train() # 训练网络。会自动加载数据，完成训练
network.evaluate() # 评价网络。在测试集上进行测试
项目的配置在setting.py里
每个网络的配置在相应的类中
网络的所有配置
"""
import logging
import numpy
from network.walk import WalkNetwork
from network.mnist import MnistNetwork
import logging

from network.walk import WalkNetwork

if __name__ == "__main__":
    # logging.debug("开始运行网络"+"="*100)
    # network = WalkNetwork()
    # network.train()
    # network.evaluate()

    pass