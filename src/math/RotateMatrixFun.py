# -*- coding=UTF-8 -*-

import numpy as np
from src.util.UtilTool import *


def rotate_3Dmatrix_180( input3Dmat ):
    logger.debug("Rotation begin ...")
    logger.debug("Input 3dimage size: " + str(input3Dmat.shape) )

    lr_flip_image = np.fliplr(input3Dmat)

    rool_lr_flip  = np.rollaxis(lr_flip_image, 2, 0)

    final_rotated_mat = np.rollaxis( np.rollaxis( np.flipud(rool_lr_flip), 1, 0), 2, 1)

    logger.debug("Final rotated matrix size: " + str(final_rotated_mat.shape) )
    return final_rotated_mat

def Test_rotate_3Dmatrix():

    x = np.arange(24).reshape(2,3,4)
    logger.debug("Test1: x is: ")
    logger.debug(str(x))
    rx = rotate_3Dmatrix_180(x)
    logger.debug("After roted:")
    logger.debug(str(rx))

    x = np.arange(18).reshape(2,3,3)
    logger.debug("Test2: x is: ")
    logger.debug(str(x))
    rx = rotate_3Dmatrix_180(x)
    logger.debug("After roted:")
    logger.debug(str(rx))

    x = np.arange(32).reshape(2,4,4)
    logger.debug("Test3: x is: ")
    logger.debug(str(x))
    rx = rotate_3Dmatrix_180(x)
    logger.debug("After roted:")
    logger.debug(str(rx))

#Test_rotate_3Dmatrix()

def rotate_4D_kernel(kernel4D, degree):
    logger.debug("4D Kernel size: " + str(kernel4D.shape) + ", degree: " + str( 90 * degree) )
    kernel_cnt, previous_connect, height, width = kernel4D.shape
    for ind in range(kernel_cnt):
        for con in range( previous_connect ):
            kernel4D[ind, con, ::] = np.rot90(kernel4D[ind,con,::], degree)

    kernel4D = np.swapaxes(kernel4D, 0, 1)
    return kernel4D

def Test_rotate_4D_kernel():
    x = np.arange(36).reshape(3,3,2,2)
    logger.debug("Test 1: x shape: " + str(x.shape))
    logger.debug("\n" + str(x))
    rx = rotate_4D_kernel(x, 2)
    logger.debug("Rotated kernel shape: " + str(rx.shape) )
    logger.debug("\n" + str(rx))

#Test_rotate_4D_kernel()
