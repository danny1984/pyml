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

# mnist_loader
import mnist_loader
import mynetwork

training_data, validate_data, test_data = mnist_loader.load_data_wrapper()
net = mynetwork.NNetwork([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

