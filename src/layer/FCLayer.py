# coding: utf-8

from LayerBase import LayerBase
from LayerType import *
from src.util.UtilTool import *
from src.factory.GlobalFactory import *
from src.factory.MathFactory import *

class FCLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(FCLayer, self).__init__(config, globalConfig)
        self.type = LayerType.FULL_CONNECT
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
        if bottom.type == LayerType.FULL_CONNECT or bottom.type == LayerType.INPUT_LAYER:
            self.default_weight_initializer(top._N, bottom._N)
        else:
            # 前面一层是卷积层 或者 池化层时候，bottom._N 是个Tensor
            bottom_N = vector_prod(bottom._N)
            self.default_weight_initializer(top._N, bottom_N)
        logger.debug("setup current_layer:[" + top.getLayerName() + "] Weight_size: "
                    + str(self._W.shape) + " Bias_size: " + str(self._B.shape))

    def forward(self, preLayer):
        super(FCLayer, self).forward()
        logger.debug("PreLayer output size: " + str(preLayer._A.shape)
                    + " current layer weight_size: " + str(self._W.shape)
                    + " current layer base_size: " + str(self._B.shape) )

        tmpReshape = preLayer._A.reshape(-1, 1)
        logger.debug("PreLayer output reshape: " + str(tmpReshape.shape))
        self._Z = np.dot(self._W, tmpReshape) + self._B
        '''
        can use different activeFunction
        '''
        self._A = sigmoid_vec(self._Z)
        logger.debug(self.getLayerName() + " output size: " + str(self._A.shape))

    def backward(self, preLayer, postLayer):
        super(FCLayer, self).backward()
        logger.debug(self.getLayerName() + " postLayer._W.transpose size: " + str(postLayer._W.transpose().shape))
        logger.debug(self.getLayerName() + " postLayer.delta size: " + str( postLayer.delta.shape ))
        logger.debug(self.getLayerName() + " self._Z size: " + str( self._Z.shape ))
        self.delta = np.dot(postLayer._W.transpose(), postLayer.delta) * sigmoid_prime_vec(self._Z)
        logger.debug(self.getLayerName() + " delta size: " + str(self.delta.shape))

        tmpReshape = preLayer._A.reshape(-1, 1)
        logger.debug(self.getLayerName() + " preLayer._A.reshape size: " + str( tmpReshape.transpose().shape ))
        logger.debug(self.getLayerName() + " self.delta_W size: " + str( self.delta_W.shape ))
        logger.debug(self.getLayerName() + " self.delta_B size: " + str( self.delta_B.shape ))
        self.delta_W = self.delta_W + np.dot(self.delta, tmpReshape.transpose())
        self.delta_B = self.delta_B + self.delta
        logger.debug(self.getLayerName() + " backward done, delta_W size: " + str(self.delta_W.shape)
                     + " delta_B size: " + str(self.delta_B.shape))

    def default_weight_initializer(self, width, length):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self._B = np.random.randn(width, 1)
        self._W = np.random.randn(width, length)/np.sqrt(width)

    def batchUpdate(self, batch_size, eta):
        logger.debug(self.getLayerName() + " batch update")
        logger.debug(self.getLayerName() + " batch_size: " + str(batch_size) + ", eta: " + str(eta))
        logger.debug(self.getLayerName() + " average W: " + str(np.average(self._W)))
        logger.debug(self.getLayerName() + " average delta_W: " + str(np.average(self.delta_W)))
        logger.debug(self.getLayerName() + " average B: " + str(np.average(self._B)))
        logger.debug(self.getLayerName() + " average delta_B: " + str(np.average(self.delta_B)))
        self._W = self._W - (float(eta)/batch_size) * self.delta_W
        self._B = self._B - (float(eta)/batch_size) * self.delta_B

    def miniBatchPrepare(self):
        self.delta_B = np.zeros(self._B.shape)
        self.delta_W = np.zeros(self._W.shape)
