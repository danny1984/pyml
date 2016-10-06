# coding: utf-8

from src.factory.GlobalFactory import *
from LayerBase import LayerBase
from LayerType import *
from src.factory.MathFactory import *
from src.util.UtilTool import *

class OutputLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(OutputLayer, self).__init__(config, globalConfig)
        self.type = LayerType.OUTPUT_LAYER
        self._N = config["num_output"]
        self._W = []
        self._B = []
        self._Z = []
        self._A = []
        self.delta = []
        self.delta_W = []
        self.delta_B = []
        self.costFunc = globals()[globalConfig["parameters"]["costType"]]()
        self._W_shape = []
        self._B_shape = []

    def setup(self, bottom, top):
        super(OutputLayer, self).setup(bottom, top)
        logger.debug("pre_layer: [" + bottom.getLayerName() + "] current_layer: [" + top.getLayerName() + "]" )

        # output 的weight 和 bias
        self.default_weight_initializer(top._N, bottom._N)
        logger.debug("setup current_layer:[" + top.getLayerName() + "] Weight_size: "
                    + str(self._W.shape) + " Bias_size: " + str(self._B.shape))

    def forward(self, preLayer):
        super(OutputLayer, self).forward()
        logger.debug("PreLayer output size: " + str(preLayer._A.shape)
                    + " current layer weight_size: " + str(self._W.shape)
                    + " current layer base_size: " + str(self._B.shape) )
        tmpReshape = preLayer._A.reshape(-1, 1)
        self._Z = np.dot(self._W, tmpReshape) + self._B
        '''
        can use different activeFunction
        '''
        self._A = sigmoid_vec(self._Z)
        logger.debug(self.getLayerName() + " output size: " + str(self._A.shape))

    ''' 对于output层来说，BP时候是特殊的，需要根据cost function 以及最终的标签确定
    '''
    def backward(self, preLayer, Y):
        super(OutputLayer, self).backward()
        # 输出层 神经元的 delta 计算方式如下: delta 定义是cost 与 输入z的导数
        self.delta = self.costFunc.delta(self._Z, self._A, Y)

        # delta 是 cost与z的导数
        # derivation ( C / W_l ) = delta_l * _A_(l-1)
        # 所以这里需要引入 prelayer，拿到前一层的 action
        logger.debug(self.getLayerName() + " delta size: " + str(self.delta.shape))
        logger.debug(self.getLayerName() + " preLayer._A.transpose size: " + str( preLayer._A.transpose().shape ))
        logger.debug(self.getLayerName() + " self.delta_W size: " + str( self.delta_W.shape ))
        logger.debug(self.getLayerName() + " self.delta_B size: " + str( self.delta_B.shape ))
        self.delta_W = self.delta_W + np.dot(self.delta, preLayer._A.transpose())
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

    def miniBatchPrepare(self):
        super(OutputLayer, self).miniBatchPrepare();
        self.delta_W = np.zeros(self._W.shape)
        self.delta_B = np.zeros(self._B.shape)

    def batchUpdate(self, batch_size, eta):
        logger.debug(self.getLayerName() + " batch update: size: " + str(batch_size) + " eta: " + str(eta) )
        self._W = self._W - (float(eta)/batch_size) * self.delta_W
        self._B = self._B - (float(eta)/batch_size) * self.delta_B

    def doCost(self, Y):
        return self.costFunc.cost(self._A, Y)
