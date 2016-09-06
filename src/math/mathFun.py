# coding: utf-8

import numpy as np
from functools import reduce

def padwithzero(vector, pad_width, iaxis, kwargs):
     vector[:pad_width[0]] = 0
     vector[-pad_width[1]:] = 0
     return vector

def vector_prod(f):
    return reduce(lambda x, y: x*y, f)

def vectorize_result(vector_size, y):
    e = np.zeros((vector_size, 1))
    e[y] = 1.0
    return e

def _im2col(image, block_size, skip=1):

    rows, cols = image.shape
    horz_blocks = cols - block_size[1] + 1
    vert_blocks = rows - block_size[0] + 1

    output_vectors = np.zeros((block_size[0] * block_size[1], horz_blocks * vert_blocks))
    itr = 0
    for v_b in xrange(vert_blocks):
        for h_b in xrange(horz_blocks):
            output_vectors[:, itr] = image[v_b: v_b + block_size[0], h_b: h_b + block_size[1]].ravel()
            itr += 1

    return output_vectors[:, ::skip]


#def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
#    # First figure out what the size of the output should be
#    N, C, H, W = x_shape
#    assert (H + 2 * padding - field_height) % stride == 0
#    assert (W + 2 * padding - field_height) % stride == 0
#    out_height = (H + 2 * padding - field_height) / stride + 1
#    out_width = (W + 2 * padding - field_width) / stride + 1
#    i0 = np.repeat(np.arange(field_height), field_width)
#    i0 = np.tile(i0, C)
#    i1 = stride * np.repeat(np.arange(out_height), out_width)
#    j0 = np.tile(np.arange(field_width), field_height * C)
#    j1 = stride * np.tile(np.arange(out_width), out_height)
#    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
#
#    return (k, i, j)
#
#def im2col_indices(x, field_height, field_width, padding=1, stride=1):
#    """ An implementation of im2col based on some fancy indexing """
#    # Zero-pad the input
#    p = padding
#    ######### This function is not workable ##############
#    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
#    ######################################################
#    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
#    cols = x_padded[:, k, i, j]
#    C = x.shape[1]
#    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
#    return cols
