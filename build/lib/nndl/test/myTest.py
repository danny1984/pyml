"""
network.py

author: dehong.gdh at alibaba-inc.com(Hangzhou)

Introduction: neural network and deep learning in action 
(website: http://neuralnetworksanddeeplearning.com/index.html)

Chapter 1,3: neural networks for handwritting digits recognizaion

"""
# Standard library
# Third-party libraries

# mnist_loader
from nndl.backcode import mynetwork
from src.common import mnist_loader

training_data, validate_data, test_data = mnist_loader.load_data_wrapper()
# chapter 1: use quadratic cost function
net = mynetwork.NNetwork([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# chapter 3: use cross-entropy cost function
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, \
#         monitor_evaluation_accuracy=True, monitor_training_cost=True)




