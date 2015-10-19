"""
network.py

author: dehong.gdh at alibaba-inc.com(Hangzhou)

Introduction: neural network and deep learning in action 
(website: http://neuralnetworksanddeeplearning.com/index.html)

Chapter 1: neural networks for handwritting digits recognizaion

"""
# Standard library
import random
# Third-party libraries
import numpy as np


class NNetwork():
    def __init__(self, sizes):
        self.num_layers = len(sizes);
        self.sizes      = sizes;
        self.biases     = [np.random.randn(y,1) for y in sizes[1:]];
        self.weights    = [np.random.randn(y,x)
                           for x,y in zip(sizes[:-1], sizes[1:]) ];
        