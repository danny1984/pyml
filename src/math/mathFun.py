# coding: utf-8

import numpy as np

def vectorize_result(vector_size, y):
    e = np.zeros((vector_size, 1))
    e[y] = 1.0
    return e


