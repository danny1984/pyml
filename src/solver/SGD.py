# coding: utf-8

import random
from src.factory.GlobalFactory import *
from src.util.UtilTool import *
from src.solver.SolverBase import *

class SGD(SolverBase):

    def __init__(self, nn):
        logger.debug("initialize SGD instance")
        self.neuronNetwork = nn
        logger.debug("initialize neuronNetwork[" + nn.neuronNetworkName + "] in SGD")

    def doOptimize(self):
        logger.debug("===== SGD begin, to optimal network[" + self.neuronNetwork.neuronNetworkName + "] ======")
        n_train_size = len(self.neuronNetwork.trainData)

        training_cost, training_accuracy = [], []
        test_cost, test_accuracy = [], []
        for epoch in xrange( self.neuronNetwork.epochs ):
            logger.info("==== epoch " + str(epoch) + " =====")
            random.shuffle(self.neuronNetwork.trainData)

            mini_batches = [ self.neuronNetwork.trainData[k:k+self.neuronNetwork.mini_batch_size]
                             for k in xrange(0, n_train_size, self.neuronNetwork.mini_batch_size) ]

            for mini_batch in mini_batches:
                self.miniBatchUpdate(mini_batch)

            if self.neuronNetwork.traceTrainingCost == 1:
                cost = self.doCost(self.neuronNetwork.trainData)
                training_cost.append(cost)
                logger.info("trace training cost: " + str(cost))

            if self.neuronNetwork.traceTestCost == 1:
                cost = self.doCost(self.neuronNetwork.testData, dataType = True)
                test_cost.append(cost)
                logger.info("trace test cost: " + str(cost))

            if self.neuronNetwork.traceTrainingAccuracy == 1:
                accuracy = self.doAccuracy(self.neuronNetwork.trainData, dataType = True)
                training_accuracy.append(accuracy)
                logger.info("trace training accuracy: " + str(accuracy))

            if self.neuronNetwork.traceTestAccuracy == 1:
                accuracy = self.doAccuracy(self.neuronNetwork.testData)
                test_accuracy.append(accuracy)
                logger.info("trace test accuracy: " + str(accuracy) )

    def doAccuracy(self, data, dataType = False):
        results = []
        for x, y in data:
            self.doForward(x)
            if dataType:
                results.append((np.argmax(self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1]._A), np.argmax(y)))
            else:
                results.append((np.argmax(self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1]._A), y))

        acc = float(sum(int(x == y) for (x, y) in results)) / len(data)
        return acc

    def doCost(self, data, dataType = False):
        cost = 0.0
        for x, y in data:
            self.doForward(x)
            if dataType: y = vectorize_result(10, y)
            # After do forward, we can get the active val, so we just put y in the function
            cost += self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1 ].doCost(y)
        return cost / len(data)

    def miniBatchUpdate(self, mini_batch):
        self.miniBatchPrepare()
        for x, y in mini_batch:
            logger.debug("Training one sample in mini_batch_update")
            self.doForward(x)
            self.doBackward(y)
        self.doBatchUpdate( self.neuronNetwork.mini_batch_size, self.neuronNetwork.eta )

    def doForward(self, x):
        logger.debug("mini_batch do forwarding: ")
        for ind in xrange(len(self.neuronNetwork.layers)):
            if ind == 0:
                self.neuronNetwork.layers[ind].forward(x)
            else:
                self.neuronNetwork.layers[ind].forward(self.neuronNetwork.layers[ind - 1])

    def doBackward(self, y):
        logger.debug("mini_batch do backforwarding: ")
        for ind in xrange(1, len(self.neuronNetwork.layers) ):
            if ind == 1:
                self.neuronNetwork.layers[-ind].backward(self.neuronNetwork.layers[- ind - 1], y)
            else:
                self.neuronNetwork.layers[-ind].backward(self.neuronNetwork.layers[- ind - 1],
                                                         self.neuronNetwork.layers[- ind + 1])

    def doBatchUpdate(self, batch_size, eta):
        for layer in self.neuronNetwork.layers:
            layer.batchUpdate(batch_size, eta)

    def miniBatchPrepare(self):
        for layer in self.neuronNetwork.layers:
            layer.miniBatchPrepare()