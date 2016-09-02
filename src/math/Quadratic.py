# coding: utf-8

import numpy as np
from src.factory.MathFactory import *

'''
    均方误差 cost: 在Deep learning中较少用到，原因是它的delta会引起梯度饱和(gradient saturate)问题
                    但是在线性回归中会经常使用 quadratic function 原因是线性回归中输出值不一定是[0,1]，
                    不需要通过做非线性 sigmod 转换
'''

class Quadratic(object):

    def __init__(self):
        self._name = "Quadratic"
        self._cost = 0

    ''' 返回激活值和目标值的cost
        activeVal, targetVal 均是 vector，例如手写分类中某个样本的的标签是[1,0,0,0,0,0,0,0,0,0]
    '''
    def cost(self, activeVal, targetVal):
        return 0.5 * np.linalg.norm( activeVal - targetVal ) ** 2

    ''' 返回最后一层的delta，用于backforward
    '''
    def delta(self, _Z, activeVal, targetVal):
        return (activeVal - targetVal) * sigmoid_prime_vec(_Z)

