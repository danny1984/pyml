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
    
    def feedforword(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a) + b);
        return a;
    
        
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)