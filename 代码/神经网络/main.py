import logging
from network.walk import WalkNetwork
from network.mnist import MnistNetwork
import logging

from network.walk import WalkNetwork

if __name__ == "__main__":
    logging.debug("开始运行网络"+"="*100)
    network = WalkNetwork()
    network.train()
    network.evaluate()