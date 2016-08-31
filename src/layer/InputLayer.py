# coding: utf-8

from LayerBase import LayerBase
from src.factory.GlobalFactory import *
from src.util.UtilTool import *

class InputLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(InputLayer, self).__init__(config, globalConfig)
        self.dataConfig = globalConfig["data"]
        self.N_      = 0    # 神经元个数，因为是输入层所以神经元个数等于数据的输入维度
        self.layerOutput  = []

    def setup(self, data, curLayer):
        super(InputLayer, self).setup("", curLayer)
        self.N_ = len(data[0][0][0])
        logger.info("输入维度 N: " + str(self.N_))

    def miniBatchPrepare(self):
        super(InputLayer, self).miniBatchPrepare()

    def forward(self, sample):
        super(InputLayer, self).forward()
        logger.info("InputLayer forward: " + str(sample.size))
        self.layerOutput = sample

    def backward(self, postLayer):
        super(InputLayer, self).backward()

