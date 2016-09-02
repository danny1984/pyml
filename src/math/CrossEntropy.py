# coding: utf-8

from src.util.UtilTool import *
from src.factory.ToolFactory import *
from src.factory.MathFactory import *

class CrossEntropy(object):

    def __init__(self):
        self._name = "CrossEntropy"
        self._cost = 0

    ''' 返回激活值和目标值的cost
        activeVal, targetVal 均是 vector，例如手写分类中某个样本的的标签是[1,0,0,0,0,0,0,0,0,0]
    '''
    def cost(self, activeVal, targetVal):
        return np.sum(np.nan_to_num(-targetVal * np.log(activeVal) - (1 - targetVal) * np.log(1 - activeVal)))

    ''' 返回最后一层的delta，用于backforward
    '''
    def delta(self, _Z, activeVal, targetVal):
        return (activeVal - targetVal)

