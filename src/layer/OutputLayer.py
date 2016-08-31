# coding: utf-8

from LayerBase import LayerBase
from src.factory.MathFactory import *
from src.util.UtilTool import *

class OutputLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(OutputLayer, self).__init__(config, globalConfig)
        self.N_ = config["num_output"]
        self.W_ = []
        self.B_ = []
        self.Z_ = []
        self.layerOutput = []

    def setup(self, bottom, top):
        super(OutputLayer, self).setup(bottom, top)
        logger.info("pre_layer: [" + bottom.getLayerName() + "] current_layer: [" + top.getLayerName() + "]" )

        # output 的weight 和 bias
        self.default_weight_initializer(top.N_, bottom.N_)
        logger.info("setup current_layer:[" + top.getLayerName() + "] Weight_size: "
                    + str(self.W_.shape) + " Bias_size: " + str(self.B_.shape))

    def forward(self, preLayer):
        super(OutputLayer, self).forward()
        logger.info("PreLayer output size: " + str(preLayer.layerOutput.shape)
                    + " current layer weight_size: " + str(self.W_.shape)
                    + " current layer base_size: " + str(self.B_.shape) )
        self.Z_ = np.dot(self.W_, preLayer.layerOutput) + self.B_
        '''
        can use different activeFunction
        '''
        self.layerOutput = sigmoid_vec(self.Z_)
        logger.info(self.getLayerName() + " output size: " + str(self.Z_.shape) + " " + str(self.layerOutput.shape))

    def backward(self, postLayer):
        super(OutputLayer, self).backward()

    def default_weight_initializer(self, width, length):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self.B_ = np.random.randn(width, 1)
        self.W_ = np.random.randn(width, length)/np.sqrt(width)

    def miniBatchPrepare(self):
        super(OutputLayer, self).miniBatchPrepare();