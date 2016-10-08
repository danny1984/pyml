# coding: utf-8

import random
import datetime
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
        optimize_start = datetime.datetime.now()
        for epoch in xrange( self.neuronNetwork.epochs ):
            logger.info("==== epoch " + str(epoch) + " =====")
            random.shuffle(self.neuronNetwork.trainData)

            epoch_start = datetime.datetime.now()
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

            epoch_end = datetime.datetime.now()
            logger.info("This epoch takes " + str(float((epoch_end - epoch_start).seconds)/60.0) + " mins!")

        optimize_end = datetime.datetime.now()
        logger.info("=== Whole optimization lasts " + str( float((optimize_end - optimize_start).seconds)/60.0 ) + " mins ===")

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
            #self.check_cnn_delta_w(x, y)
            #self.check_delta_w(x, y)

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

    def check_cnn_delta_w(self, x, y):
        h = 0.0001
        kernel_ind        = 0
        kernel_connect    = 0
        weight_height_ind = 0
        weight_width_ind  = 0

        # 想要check的layer
        ind = 1
        # BP 得到的 delta_W delta_B
        logger.info("Gradient check: from network" + str(self.neuronNetwork.layers[ind].delta_tmp_W[kernel_ind, kernel_connect, weight_height_ind,weight_width_ind]))
        derivation_w_from_bp = self.neuronNetwork.layers[ind].delta_tmp_W[kernel_ind, kernel_connect, weight_height_ind, weight_width_ind]

        # Gradient check - W
        logger.info("Gradient check: original layer weight " + str(self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind, weight_width_ind]) )
        self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind, weight_width_ind] = \
            self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind,weight_width_ind] + h
        logger.info("Gradient check: add new h the layer weight is " + str(self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind,weight_width_ind]) )
        self.doForward(x)
        cost_plus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1 ].doCost(y)
        logger.info("Gradient check: plus h cost is " + str(cost_plus_h) )

        self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind,weight_width_ind] = \
            self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind,weight_width_ind] - 2.0 * h
        self.doForward(x)
        cost_minus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1].doCost(y)
        logger.info("Gradient check: minus h cost is " + str(cost_minus_h) )
        derivation_of_weight = (cost_plus_h - cost_minus_h)/(2.0*h)

        # check 是否相同
        logger.info("Gradient check: BP delta_W: " + str(derivation_w_from_bp) + ", gradient check: " + str(derivation_of_weight) )
        if np.fabs((derivation_w_from_bp - derivation_of_weight)) < 0.001:
            logger.info("Gradient check weight sucessfully")
        else:
            logger.info("Error in gradient weight check in BP")
            exit(-1)
        self.neuronNetwork.layers[ind]._W[kernel_ind,kernel_connect, weight_height_ind, weight_width_ind] = \
            self.neuronNetwork.layers[ind]._W[kernel_ind, kernel_connect, weight_height_ind, weight_width_ind] + h

        # Gradient check - B
        derivation_b_from_bp = self.neuronNetwork.layers[ind].delta_tmp_B[kernel_ind]

        # check B
        logger.info("Gradient check: original layer bias " + str(self.neuronNetwork.layers[ind]._B[kernel_ind]))
        self.neuronNetwork.layers[ind]._B[kernel_ind] = self.neuronNetwork.layers[ind]._B[kernel_ind] + h
        logger.info("Gradient check: add new h to the layer bias " + str(self.neuronNetwork.layers[ind]._B[kernel_ind]) )
        self.doForward(x)
        cost_plus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1 ].doCost(y)
        logger.info("Gradient check: plus h cost is " + str(cost_plus_h))
        self.neuronNetwork.layers[ind]._B[kernel_ind] = self.neuronNetwork.layers[ind]._B[kernel_ind] - 2.0 * h
        self.doForward(x)
        cost_minus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1 ].doCost(y)
        logger.info("Gradient check: minus h cost is " + str(cost_minus_h) )
        derivation_of_bias = (cost_plus_h - cost_minus_h) / (2.0 * h)

        # check 是否相同
        logger.info("Gradient check: BP delta_B: " + str(derivation_b_from_bp) + ", gradient check: " + str(derivation_of_bias) )
        if np.fabs( (derivation_b_from_bp - derivation_of_bias)) < 0.001:
            logger.info("Gradient check bias sucessfully")
        else:
            logger.info("Error in gradient bias check in BP")
            exit(-1)

        self.neuronNetwork.layers[ind]._B[kernel_ind] = self.neuronNetwork.layers[ind]._B[kernel_ind] + h


    def check_delta_w(self, x, y):
        h = 0.0001
        weight_height_ind = 0
        weight_width_ind  = 155
        # 想要check的layer
        ind = 1
        logger.info("Gradient check: from network" + str(self.neuronNetwork.layers[ind].delta_W[weight_height_ind,weight_width_ind]))
        derivation_from_bp = self.neuronNetwork.layers[ind].delta_W[weight_height_ind, weight_width_ind]
        logger.info("Gradient check: original layer weight " + str(self.neuronNetwork.layers[ind]._W[weight_height_ind, weight_width_ind]) )
        self.neuronNetwork.layers[ind]._W[weight_height_ind, weight_width_ind] = self.neuronNetwork.layers[ind]._W[weight_height_ind,weight_width_ind] + h
        logger.info("Gradient check: add new h the layer weight is " + str(self.neuronNetwork.layers[ind]._W[weight_height_ind,weight_width_ind]) )
        self.doForward(x)
        cost_plus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1 ].doCost(y)
        logger.info("Gradient check: plus h cost is " + str(cost_plus_h) )
        self.neuronNetwork.layers[ind]._W[weight_height_ind,weight_width_ind] = self.neuronNetwork.layers[ind]._W[weight_height_ind,weight_width_ind] - 2 * h
        self.doForward(x)
        cost_minus_h = self.neuronNetwork.layers[ len(self.neuronNetwork.layers) - 1].doCost(y)
        logger.info("Gradient check: minus h cost is " + str(cost_minus_h) )
        derivation_of_weight = (cost_plus_h - cost_minus_h)/(2.0*h)
        if np.fabs((derivation_from_bp - derivation_of_weight)) < 0.001:
            logger.info("Gradient check sucessfully")
        else:
            logger.info("Error in gradient check in BP")
            exit(-1)
        self.neuronNetwork.layers[ind]._W[0,0] = self.neuronNetwork.layers[ind]._W[0,0] + h
