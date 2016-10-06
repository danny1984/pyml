# -*- coding=utf-8 -*-

import numpy as np
from src.util.UtilTool import *

def  padding_3Dimage(input_3Dimage, padding_size):
    logger.debug("Padding begin ....")
    logger.debug("Input 3D image size: " + str(input_3Dimage.shape))

    (channel_cnt, height, width) = input_3Dimage.shape
    final_returned_padded_image = np.zeros((1, height + 2*padding_size, width + 2 * padding_size))
    logger.debug("Final return image size: " + str(final_returned_padded_image.shape))
    for c in range(channel_cnt):
        pad_matric = input_3Dimage[c,::]
        pad_matric = np.pad(pad_matric, padding_size, 'constant').reshape(1, height + 2*padding_size, width + 2*padding_size)
        final_returned_padded_image = np.vstack( (final_returned_padded_image, pad_matric) )

    logger.debug("Padding input 3Dimage size: " + str(final_returned_padded_image.shape))
    #final_returned_padded_image = np.rollaxis(final_returned_padded_image, 2, 0)
    final_returned_padded_image = final_returned_padded_image[1:,::]
    logger.debug("After padding input 3Dimage size: " + str(final_returned_padded_image.shape))
    return final_returned_padded_image


def Test_padding_3D_image():
    t1 = np.arange(24).reshape(2, 3, 4)
    tt = padding_3Dimage(t1, 2)
    logger.debug(str(tt))

#Test_padding_3D_image()
