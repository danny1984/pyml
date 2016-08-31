# coding: utf-8

import random
from src.factory.GlobalFactory import *
from src.util.UtilTool import *
from src.solver.SolverBase import *

class SGD(SolverBase):

    def __init__(self, nn):
        logger.info("initialize SGD instance")
        self.neuronNetwork = nn
        logger.info("initialize neuronNetwork[" + nn.neuronNetworkName + "] in SGD")

    def doOptimize(self):
        logger.info("===== SGD begin, to optimal network[" + self.neuronNetwork.neuronNetworkName + "] ======")
        n_train_size = len(self.neuronNetwork.trainData)

        for epoch in xrange( self.neuronNetwork.epochs ):
            logger.info("==== epoch " + str(epoch) + " =====")
            random.shuffle(self.neuronNetwork.trainData)

            mini_batches = [ self.neuronNetwork.trainData[k:k+self.neuronNetwork.min_batch_size]
                             for k in xrange(0, n_train_size, self.neuronNetwork.min_batch_size) ]

            for mini_batch in mini_batches:
                self.miniBatchUpdate(mini_batch)


    def miniBatchUpdate(self, mini_batch):

        self.miniBatchPrepare()
        for x, y in mini_batch:
            logger.info("Training one sample in mini_batch_update")
            self.doForward(x)

    def doForward(self, x):
        logger.info("mini_batch do forwarding: ")
        for ind in xrange(len(self.neuronNetwork.layers)):
            if ind == 0:
                self.neuronNetwork.layers[ind].forward(x)
            else:
                self.neuronNetwork.layers[ind].forward(self.neuronNetwork.layers[ind - 1])


    def miniBatchPrepare(self):
        for layer in self.neuronNetwork.layers:
            layer.miniBatchPrepare()