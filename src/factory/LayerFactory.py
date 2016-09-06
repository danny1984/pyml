# coding:utf-8

# 新增加的layer只要放到这里就好了
# 调用方只要import 文件就好了
# e.g. from src.layer.LayerInclude import *

from src.layer.InputLayer import InputLayer
from src.layer.FCLayer import FCLayer
from src.layer.ConvolutionLayer import  ConvolutionLayer
from src.layer.MaxPooling import  MaxPooling
from src.layer.OutputLayer import OutputLayer