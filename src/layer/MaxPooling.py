# -*- coding: UTF-8 -*-

from src.util.UtilTool import *
from src.factory.MathFactory import *
from LayerBase import LayerBase
from LayerType import *
import numpy as np

class MaxPooling(LayerBase):

    def __init__(self, config, globalConfig):
        super(MaxPooling, self).__init__(config, globalConfig)
        self.type        = LayerType.MAX_POOLING
        self.kernel_size = config["pooling_parameters"]["kernel_size"]
        self.stride      = config["pooling_parameters"]["stride"]
        self.kernel_cnt  = 0     #kernel_cnt 是卷积层output的channel数
        self.pad         = 0
        self._A          = 0
        self._N          = []
        # _Z dimension
        self._Z          = []
        self.delta       = []

        # Pooling Mask 用于记录pooling时候的位置，比如maxpooling时候Kernel中最大元素的位置，
        # meanpooling则表示所有元素的位置
        # Pooling Mask 在 Forward时候生成的，即记录下pooling操作的位置
        #              在 Backward时候使用，主要用途是与 残差delta进行矩阵相乘，进行反向传播
        # Pooling Mask 的作用, 一定程度上，可以看做是 Pooling 层和前一层的链接
        self._PoolingMask = []

        # Pooling 层不需要 _W, _B, 所以也没有delta_W, delta_B

    def setup(self, bottom, top):
        super(MaxPooling, self).setup(bottom, top)
        self._N = self.calOutputSize(bottom)
        logger.debug(self.getLayerName() + " output size: " + str(self._N))

    def forward(self, preLayer):
        super(MaxPooling, self).forward()
        # Pooling Mask 初始化
        logger.debug("initialize pooling mask matrix, size: " + str(preLayer._A.shape))
        self._PoolingMask = np.zeros(preLayer._A.shape)

        channel, height, width = preLayer._A.shape
        # main loop
        self._Z = np.zeros(self._N, np.float32)
        logger.debug(self.getLayerName() + " channel: " + str(channel)
                     + ", height: " + str(height) + ", stride: " + str(self.stride)
                     + ", width:  " + str(width)  + ", stride: " + str(self.stride))
        for c in range(channel):
            for h in range(0, height, self.stride):
                for w in range(0, width, self.stride):
                    # pooling section
                    h_start = h
                    h_end   = min(h + self.kernel_size, height)
                    w_start = w
                    w_end   = min(w + self.kernel_size, width)

                    # output neuron position
                    out_h_pos = h/self.stride
                    out_w_pos = w/self.stride

                    # max value
                    max_in_pooling_section = 0
                    # max value index
                    mask_max_height_index = h_start
                    mask_max_width_index  = w_start

                    # 迭代pooling区域找到最大值以及对应的坐标
                    for iter_h in xrange(h_start, h_end):
                        for iter_w in xrange(w_start, w_end):
                            if preLayer._A[c][iter_h][iter_w] > max_in_pooling_section:
                                max_in_pooling_section = preLayer._A[c][iter_h][iter_w]
                                mask_max_height_index  = iter_h
                                mask_max_width_index   = iter_w

                    # 设置输出值，以及最大在原始pooling区块的位置
                    self._Z[c][out_h_pos][out_w_pos] = max_in_pooling_section
                    self._PoolingMask[c][mask_max_height_index][mask_max_width_index] = 1

        self._A = self._Z
        logger.debug(self.getLayerName() + " output size: " + str(self._A.shape))

    def backward(self, preLayer, postLayer):
        super(MaxPooling, self).backward()
        logger.debug(self.getLayerName() + " postLayer._W size: " + str(postLayer._W.shape))
        logger.debug(self.getLayerName() + " postLayer.delta size: " + str( postLayer.delta.shape ))
        logger.debug(self.getLayerName() + " self._Z size: " + str( self._Z.shape ))

        # 前面一层是FC 或者Output的话， 直接将误差回传
        if postLayer.type == LayerType.FULL_CONNECT or postLayer.type == LayerType.OUTPUT_LAYER :
            # 这里不需要乘以 self._Z 的导数，因为max_pooling self._A = self._Z
            #self.delta = np.dot(postLayer._W.transpose(), postLayer.delta).reshape(self._Z.shape) * sigmoid_prime_vec(self._Z)
            self.delta = np.dot(postLayer._W.transpose(), postLayer.delta).reshape(self._N)
        elif postLayer.type == LayerType.CONV:
            # 当前面一层是卷积层时，pooling残差 = ( padding后的卷积层残差  卷积上  rotated180度的卷积核 ) 叉积 （pooling层_Z的导数）
            # 1. pad 卷积层的残差，pad_size = kernel_size - 1
            logger.debug(self.getLayerName() + " to pad postLayer delta, size: " + str(postLayer.delta.shape) + ", padding_size: " + str(postLayer.kernel_size - 1) )
            conv_delta_padded = padding_3Dimage(postLayer.delta, postLayer.kernel_size - 1)
            logger.debug(self.getLayerName() + " postLayer delta padding size: " + str(conv_delta_padded.shape) )

            # 2. rotate postLayer.kernel with 180 degree
            logger.debug(self.getLayerName() + " postLayer _W size: " + str(postLayer._W.shape) + ", going to rotate this postLayer._W")
            # *****************  一定要注意同时要进行kernel组的 rotation **********************
            rotated_kernel = rotate_4D_kernel(postLayer._W, 2)
            logger.debug(self.getLayerName() + " rotated_kernel size: " + str(rotated_kernel.shape))

            # 3. 卷积  padded残差 和 rotated_kernel
            conv_postLayerDelta_kernel = Convolution3D(conv_delta_padded, rotated_kernel)
            logger.debug(self.getLayerName() + " cov_postLayerDelta_kernel size: " + str(conv_postLayerDelta_kernel.shape) )
            # 4. 叉积， maxpooling 没有使用任何非线性函数，直接输出，所以导数是1，最后变形
            self.delta = conv_postLayerDelta_kernel.transpose().reshape(self._N)

        logger.debug(self.getLayerName() + " delta size: " + str(self.delta.shape))

        # 由于没有使用非线性函数，以及缩放因子和bias，所以不用进行 delta_W 以及 delta_B 的更新
        logger.debug(self.getLayerName() + " backward done")

    def calOutputSize(self, bottom):
        logger.debug("calculate the output")
        output_size = (bottom._N[1] - self.kernel_size + 2 * self.pad )/self.stride + 1
        return [bottom._N[0], output_size, output_size]

    def miniBatchPrepare(self):
        super(MaxPooling, self).miniBatchPrepare()
        # pooling 层没有_W _B

    def batchUpdate(self, batch_size, eta):
        logger.debug(self.getLayerName() + " batch update")

