# coding: utf-8

from LayerBase import LayerBase
from src.util.UtilTool import *
from src.factory.GlobalFactory import *
from src.factory.MathFactory import *

class FCLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(FCLayer, self).__init__(config, globalConfig)
        self._N = config["num_output"]
        self._W = []
        self._B = []
        self.delta = []
        self.delta_W = []
        self.delta_B = []
        self._Z = []
        self._A = []

    def setup(self, bottom, top):
        super(FCLayer, self).setup(bottom, top)
        logger.debug("pre_layer: [" + bottom.getLayerName() + "] current_layer: [" + top.getLayerName() + "]" )

        # FC的weight 和 bias
        self.default_weight_initializer(top._N, bottom._N)
        logger.debug("setup current_layer:[" + top.getLayerName() + "] Weight_size: "
                    + str(self._W.shape) + " Bias_size: " + str(self._B.shape))

    def forward(self, preLayer):
        super(FCLayer, self).forward()
        '''
        logger.debug("PreLayer output size: " + str(preLayer._A.shape)
                    + " current layer weight_size: " + str(self._W.shape)
                    + " current layer base_size: " + str(self._B.shape) )
        '''
        self._Z = np.dot(self._W, preLayer._A) + self._B
        '''
        can use different activeFunction
        '''
        self._A = sigmoid_vec(self._Z)
        logger.debug(self.getLayerName() + " output size: " + str(self._Z.shape) + " " + str(self._A.shape))

    def backward(self, preLayer, postLayer):
        super(FCLayer, self).backward()
        self.delta = np.dot(postLayer._W.transpose(), postLayer.delta) * sigmoid_prime_vec(self._Z)

        self.delta_W = self.delta_W + np.dot(self.delta, preLayer._A.transpose())
        self.delta_B = self.delta_B + self.delta

    def default_weight_initializer(self, width, length):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self._B = np.random.randn(width, 1)
        self._W = np.random.randn(width, length)/np.sqrt(width)

    def miniBatchPrepare(self):
        self.delta_B = np.zeros(self._B.shape)
        self.delta_W = np.zeros(self._W.shape)
