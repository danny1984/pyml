# coding: utf-8

from LayerBase import LayerBase
from src.factory.GlobalFactory import *
from src.util.UtilTool import *
from LayerType import *

class InputLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(InputLayer, self).__init__(config, globalConfig)
        self.type       = LayerType.INPUT_LAYER
        self.dataConfig = globalConfig["data"]
        self._N  = 0    # 神经元个数，因为是输入层所以神经元个数等于数据的输入维度
        self._A  = []
        self.reshape = 0
        if config.has_key("reshape"):
            logger.debug("has reshape parameter")
            self.reshape = config["reshape"]["flag"]
            self.reshapeChannel = config["reshape"]["channel"]
            self.reshapeWidth   = config["reshape"]["width"]
            self.reshapeHeight  = config["reshape"]["height"]
            logger.debug("Reshape channel: " + str(self.reshapeChannel) + " width:" + str(self.reshapeWidth) + ", height:" + str(self.reshapeHeight))

    def setup(self, data, curLayer):
        super(InputLayer, self).setup("", curLayer)
        self._N = len(data[0][0][0])
        if self.reshape == 1:
            self._N = [self.reshapeChannel, self.reshapeHeight, self.reshapeWidth]
        logger.debug("输入维度 N: " + str(self._N))

    def miniBatchPrepare(self):
        super(InputLayer, self).miniBatchPrepare()

    def forward(self, sample):
        super(InputLayer, self).forward()
        logger.debug("InputLayer forward: " + str(sample.size))
        self._A = sample
        if self.reshape == 1:
            self._A = self._A.reshape(self.reshapeChannel, self.reshapeHeight, self.reshapeWidth)
        logger.debug("InputLayer forward end ")

    def backward(self, postLayer):
        super(InputLayer, self).backward()
        logger.debug(self.getLayerName() + " backwarding now")

    def batchUpdate(self, batch_size, eta):
        logger.debug(self.getLayerName() + " batch update")
