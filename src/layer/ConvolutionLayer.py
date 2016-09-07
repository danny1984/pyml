# coding = utf-8

from LayerBase import LayerBase
from src.util.UtilTool import *
from src.math.mathFun import _im2col
import numpy as np

class ConvolutionLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(ConvolutionLayer, self).__init__(config, globalConfig)
        self.kernel_size = config["conv_parameters"]["kernel_size"]
        self.stride      = config["conv_parameters"]["stride"]
        self.kernel_cnt  = config["conv_parameters"]["kernel_cnt"]
        self.pad         = 0
        self._A          = 0
        self._N          = []

    def setup(self, bottom, top):
        super(ConvolutionLayer, self).setup(bottom, top)
        self._N = self.calOutputSize(bottom)
        logger.debug(self.getLayerName() + " output size: " + str(self._N))


    def forward(self, preLayer):
        super(ConvolutionLayer, self).forward()
        logger.debug("======================================")
        x = np.arange(9).reshape(3,3)
        logger.debug(x)
        x_col = _im2col(x, (2,2))
        logger.debug(x_col)
        logger.debug("======================================")

    def backward(self, postLayer):
        super(ConvolutionLayer, self).backward()

    def calOutputSize(self, bottom):
        logger.debug("calculate the output")
        output_size = (bottom._N[1] - self.kernel_size + 2 * self.pad )/self.stride + 1
        return [self.kernel_cnt, output_size, output_size]

    def miniBatchPrepare(self):
        super(ConvolutionLayer, self).miniBatchPrepare()
