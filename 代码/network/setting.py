import os
import logging

# 配置日志配置同时输出到屏幕和日志文件
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("log.txt",encoding="utf8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 通用环境相关
DATA_PATH = "./data"  # 所有数据存放路径
DATA0_PATH = os.path.join(DATA_PATH, "data0")
MODELS_PATH = "./network/model"  # 模型位置
MNIST_DATA_FULL_PATH = os.path.join(DATA_PATH, "mnist/mnist.pkl.gz")  # mnist数据
