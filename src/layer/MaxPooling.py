# coding: utf-8

from src.util.UtilTool import *
from LayerBase import LayerBase

class MaxPooling(LayerBase):

    def __init__(self, config, globalConfig):
        super(MaxPooling, self).__init__(config, globalConfig)
        self.kernel_size = config["pooling_parameters"]["kernel_size"]
        self.stride      = config["pooling_parameters"]["stride"]
        self.kernel_cnt  = 0     #kernel_cnt 是卷积层output的channel数
        self.pad         = 0
        self._A          = 0
        self._N          = []

    def setup(self, bottom, top):
        super(MaxPooling, self).setup(bottom, top)
        self._N = self.calOutputSize(bottom)
        logger.debug(self.getLayerName() + " output size: " + str(self._N))

    def forward(self, preLayer):
        super(MaxPooling, self).forward()

    def backward(self, postLayer):
        super(MaxPooling, self).backward()

    def calOutputSize(self, bottom):
        logger.debug("calculate the output")
        output_size = (bottom._N[1] - self.kernel_size + 2 * self.pad )/self.stride + 1
        return [bottom._N[0], output_size, output_size]

    def miniBatchPrepare(self):
        super(MaxPooling, self).miniBatchPrepare()
