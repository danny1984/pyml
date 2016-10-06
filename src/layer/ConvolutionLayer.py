# coding = utf-8

from LayerBase import LayerBase
from LayerType import *
from src.util.UtilTool import *
from src.math.mathFun import im2col
from src.factory.MathFactory import  *
import numpy as np

class ConvolutionLayer(LayerBase):

    def __init__(self, config, globalConfig):
        super(ConvolutionLayer, self).__init__(config, globalConfig)
        self.type        = LayerType.CONV
        self.kernel_size = config["conv_parameters"]["kernel_size"]
        self.stride      = config["conv_parameters"]["stride"]
        self.kernel_cnt  = config["conv_parameters"]["kernel_cnt"]
        self.pad         = 0
        self._A          = 0
        self._N          = []
        """
        _W dimension  :   kernel_size * preLayer_output_channel * kernel_height * kernel_width
                          kernel_size/kernel_height/kernel_width is easy to understand
                          preLayer_output_channel represents the previous layers which the current kernel will connect to.
                              in LeNet this is the connections between CNN layers.
        """
        self._W          = []
        # _B dimension  :   kernel_cnt
        self._B          = []
        # _Z dimension
        self._Z          = []

        # delta_W
        self.delta_W    = []
        self.delta_B    = []

    def setup(self, bottom, top):
        super(ConvolutionLayer, self).setup(bottom, top)

        logger.debug(self.getLayerName() + " preLayer output size: " + str(bottom._N) )
        logger.debug(self.getLayerName() + " cnn kernel_size: " + str(self.kernel_size))
        logger.debug(self.getLayerName() + " cnn kernel_cnt: " + str(self.kernel_cnt) )
        self.default_weight_initializer(bottom._N[0], self.kernel_size, self.kernel_cnt)
        logger.debug(self.getLayerName() + " _W size: " + str(self._W.shape))
        logger.debug(self.getLayerName() + " _B size: " + str(self._B.shape))

        logger.debug(self.getLayerName() + " preLayer output size: " + str(bottom._N))
        self._N = self.calOutputSize(bottom)
        logger.debug(self.getLayerName() + " output_size: " + str(self._N))

    def forward(self, preLayer):
        super(ConvolutionLayer, self).forward()
        logger.debug(self.getLayerName() + " preLayer is " + preLayer.getLayerName() )
        logger.debug(self.getLayerName() + " preLayer _A size " + str(preLayer._A.shape) )
        logger.debug(self.getLayerName() + " preLayer _N is " + str(preLayer._N) )

        #logger.debug(self.getLayerName() + " input im2col ....")
        #input_col = im2col(preLayer._A, (self.kernel_size, self.kernel_size))
        #logger.debug(self.getLayerName() + " input im2col size: " + str(input_col.shape) )

        #logger.debug(self.getLayerName() + " kernel im2col ....")
        #kernel_cnt, previous_output_channel, kernel_height, kernel_width = self._W.shape
        #ker_col_tmp = np.zeros((previous_output_channel*kernel_height*kernel_width, 1))
        #for ind in xrange(kernel_cnt):
        #    ker = self._W[ind,::]
        #    ker_im2col = im2col(ker, (self.kernel_size, self.kernel_size))
        #    ker_col_tmp = np.concatenate( (ker_col_tmp, ker_im2col.transpose()), axis=1)

        #ker_col = ker_col_tmp[:,1:]
        #logger.debug(self.getLayerName() + " kernel col size: " + str(ker_col.shape))

        #logger.debug(self.getLayerName() + " inpu_col size: " + str(input_col.shape))
        #logger.debug(self.getLayerName() + " ker_col size: " + str(ker_col.shape))
        #logger.debug(self.getLayerName() + " _B size: " + str(self._B.shape))
        #self._Z = np.dot(input_col, ker_col) + self._B
        #logger.debug(self.getLayerName() + " dot size: " + str(np.dot(input_col, ker_col).shape) )
        #logger.debug(self.getLayerName() + " self._Z size: " + str(self._Z.shape) )
        logger.debug(self.getLayerName() + " _B size: " + str(self._B.shape))
        logger.debug(self.getLayerName() + " _W size " + str(self._W.shape) )
        self._Z = Convolution3D_with_Bias(preLayer._A, self._W, self._B)
        logger.debug(self.getLayerName() + " Z size: " + str(self._Z.shape) )
        self._A = sigmoid_vec( self._Z )
        logger.debug(self.getLayerName() + " A size: " + str(self._A.shape))

    def backward(self, preLayer, postLayer):
        super(ConvolutionLayer, self).backward()

        if postLayer.type == LayerType.MAX_POOLING:
            logger.debug(self.getLayerName() + " postLayer._PoolingMask size: " + str(postLayer._PoolingMask.shape))
            logger.debug(self.getLayerName() + " postLayer.delta size: " + str( postLayer.delta.shape )
                         + " and postLayer kernel_size: " + str(postLayer.kernel_size) )
            postLayer_delta_kron = np.kron( postLayer.delta, np.ones((postLayer.kernel_size, postLayer.kernel_size)) )
            logger.debug(self.getLayerName() + " postLayer.delta kron product with kernel_size: " + str(postLayer_delta_kron.shape) )
            logger.debug(self.getLayerName() + " self._Z size: " + str( self._Z.shape ))
            self.delta = postLayer._PoolingMask * postLayer_delta_kron * sigmoid_prime_vec(self._Z)
            logger.debug(self.getLayerName() + " self.delta size: " + str(self.delta.shape) )
        else:
            logger.fatal("***** Fatal: the layer after the " + self.getLayerName() + " convolution layer is not pooling layer!!!!!!!!!! ")

        logger.debug(self.getLayerName() + " to update delta_W ....")
        logger.debug(self.getLayerName() + " delta_W size: " + str(self.delta_W.shape) )
        logger.debug(self.getLayerName() + " delta_B size: " + str(self.delta_B.shape))
        logger.debug(self.getLayerName() + " preLayer._A size: " + str(preLayer._A.shape))
        delta_output_channel_cnt, delta_height, delta_width = self.delta.shape
        pre_A_channel_cnt, pre_A_height, pre_A_width        = preLayer._A.shape
        for delta_channel_ind in range(delta_output_channel_cnt):
            for pre_A_channel_ind in range(pre_A_channel_cnt):
                #logger.debug("preLayer._A size:" + str(preLayer._A[pre_A_channel_ind,:].shape) + " self.delta size: " + str(self.delta[delta_channel_ind,:].shape) + " self.delta_W size:" + str(self.delta_W[delta_channel_ind, pre_A_channel_ind,::].shape)  )
                self.delta_W[delta_channel_ind, pre_A_channel_ind,::] = self.delta_W[delta_channel_ind, pre_A_channel_ind, ::] \
                                                                        + Convolution2D( preLayer._A[pre_A_channel_ind,:], self.delta[delta_channel_ind,:] )
            self.delta_B[delta_channel_ind] = self.delta_B[delta_channel_ind] + np.sum(self.delta[delta_channel_ind,:])

        logger.debug(self.getLayerName() + " backward done")

    def default_weight_initializer(self, preLayer_output, kernel_size, kernel_cnt):
        self._W = np.random.randn(kernel_cnt, preLayer_output, kernel_size, kernel_size)/np.sqrt( preLayer_output * kernel_cnt * kernel_size * kernel_size)
        #self._W = np.random.randn(kernel_cnt, preLayer_output, kernel_size, kernel_size)
        self._B = np.random.randn(kernel_cnt)

    def calOutputSize(self, bottom):
        logger.debug("calculate the output")
        output_size = (bottom._N[1] - self.kernel_size + 2 * self.pad )/self.stride + 1
        return [self.kernel_cnt, output_size, output_size]

    def miniBatchPrepare(self):
        super(ConvolutionLayer, self).miniBatchPrepare()
        self.delta_W = np.zeros(self._W.shape)
        self.delta_B = np.zeros(self._B.shape)

    def batchUpdate(self, batch_size, eta):
        logger.debug(self.getLayerName() + " batch update")
        logger.debug(self.getLayerName() + " batch_size: " + str(batch_size) + ", eta: " + str(eta))
        logger.debug(self.getLayerName() + " average W: " + str(np.average(self._W)))
        logger.debug(self.getLayerName() + " average delta_W: " + str(np.average(self.delta_W)))
        logger.debug(self.getLayerName() + " average B: " + str(np.average(self._B)))
        logger.debug(self.getLayerName() + " average delta_B: " + str(np.average(self.delta_B)))
        self._W = self._W - (float(eta)/batch_size) * self.delta_W
        self._B = self._B - (float(eta)/batch_size) * self.delta_B

