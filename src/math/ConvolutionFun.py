# -*- coding: UTF-8 -*-

import numpy as np
from src.util.UtilTool import *
from src.math.mathFun import im2col
from src.math.mathFun import _im2col

'''
Description: 卷积操作
    inputMap3D: narray类型，表示输入的图片，三个维度分别表示 channel, height, width
    kernel4D:   narray类型，表示输入的kernel，四个维度分别表示 kernel_cnt, connection_with_previous_output_map, kernel_hegith, kernel_width
'''
def Convolution3D( inputMap3D, kernel4D ):
    #logger.debug("Convolution Function begin")
    kernel_cnt, previous_output_channel, kernel_height, kernel_width = kernel4D.shape
    kernel_size = (kernel_height, kernel_width)
    #logger.debug("Kernel size: " + str( kernel_size) + ", and to do input_map3d_im2col ....")
    input_col = im2col(inputMap3D, kernel_size )
    #logger.debug("After im2col inputmap3D size: " + str(input_col.shape))

    #logger.debug("To do kernel im2col....")
    #logger.debug("Kernel size: " + str(kernel4D.shape))
    ker_col_tmp = np.zeros((previous_output_channel*kernel_height*kernel_width, 1))
    for ind in xrange(kernel_cnt):
        ker = kernel4D[ind,::]
        ker_im2col = im2col(ker, kernel_size)
        ker_col_tmp = np.concatenate( (ker_col_tmp, ker_im2col.transpose()), axis=1)

    ker_col = ker_col_tmp[:,1:]
    #logger.debug("After im2col kernel col size: " + str(ker_col.shape))

    #logger.debug("Inpu_col size: " + str(input_col.shape))
    #logger.debug("Ker_col size: " + str(ker_col.shape))
    input_channel, input_height, input_width = inputMap3D.shape
    output_height = input_height - kernel_height + 1
    output_width  = input_width  - kernel_width  + 1
    ret = np.dot(input_col, ker_col).transpose().reshape(kernel_cnt, output_height, output_width)
    return ret


'''
带有bias卷积
'''
def Convolution3D_with_Bias(inputMap3D, kernel, bias):
    conv = Convolution3D(inputMap3D, kernel)
    conv_kernel_cnt, conv_height, conv_width = conv.shape
    for c in range(conv_kernel_cnt):
        bias_kron = np.kron(bias[c], np.ones((conv_height, conv_width)) )
        conv[c,:] = conv[c,:] + bias_kron
    return conv

'''
Description: 卷积操作
    inputMap2D: narray 类型，表示输入图片，两个维度是height，width
    kernel2D:   narray 类型，表示卷积核
'''
def Convolution2D(inputMap2D, kernel2D):
    #logger.debug("Convolution Function begin....")
    (kernel_height, kernel_width) = kernel2D.shape
    #logger.debug("Kernel size: " + str(kernel_height) + "," + str(kernel_width) )
    input_col = _im2col(inputMap2D, (kernel_height, kernel_width) )
    #logger.debug("After im2col input2D size: " + str(input_col.shape) )

    #logger.debug("To do kernel im2col....")
    #logger.debug("Kernel size:" + str(kernel2D.shape) )
    ker_im2col = _im2col(kernel2D, (kernel_height, kernel_width))
    #logger.debug("After im2col kernel size: " + str(ker_im2col.shape) )

    result = np.dot(input_col.transpose(), ker_im2col)
    input_height, input_width = inputMap2D.shape
    output_height = input_height - kernel_height + 1
    output_width  = input_width  - kernel_width  + 1
    return result.reshape(output_height, output_width)

def Test_convolution2D():
    x   = np.arange(16).reshape(4,4)
    logger.debug(x)
    ker = np.arange(4).reshape(2,2)
    logger.debug(ker)
    ret = Convolution2D(x, ker)
    logger.debug(ret)

#Test_convolution2D()

#def Convolution4D_New(input4DMap, kernel):
#    logger.debug(ret)

def Test_convolution3D():
    x = np.arange(18).reshape(2,3,3)
    logger.debug(x)
    ker = np.arange(24).reshape(3,2,2,2)
    logger.debug(ker)
    ret = Convolution3D(x, ker)
    logger.debug(ret)
    bias = np.ones((3,1))
    ret = Convolution3D_with_Bias(x, ker, bias)
    logger.debug(ret)

#Test_convolution3D()

