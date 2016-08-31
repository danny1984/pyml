# coding: utf-8

import sgmllib

from src.factory.GlobalFactory import *
from src.util.UtilTool import *

class SGMLNN(object):

    def __init__(self, jsonConfig):
        logger.info("SGML NN initialize begin")
        self.globalConfig = jsonConfig
        self.neuronNetworkName = jsonConfig["nn_name"]
        # 数据初始化
        logger.info("Initializing data ....")
        self.dataConfig = self.globalConfig["data"]
        self.trainData, self.validationData, self.testData = load_data_wrapper(self.dataConfig["path"])
        self.data = [self.trainData, self.validationData, self.testData]
        logger.info("training sample size: " + str(len(self.trainData[0])))
        logger.info("validation sample size: " + str(len(self.validationData[0])))
        logger.info("testing sample size: " + str(len(self.testData[0])))
        logger.info("input image size: " + str(len(self.trainData[0][0])))

        # 参数初始化
        self.parameterConfig = self.globalConfig["parameters"]
        self.min_batch_size  = self.parameterConfig["min_batch_size"]
        self.solverName      = self.parameterConfig["solver"]
        self.eta             = self.parameterConfig["eta"]
        self.epochs          = self.parameterConfig["epochs"]
        self.parameter = [self.min_batch_size, self.solverName, self.eta, self.epochs]
        logger.info("Parameters: " + ",".join([str(x) for x in self.parameter]))

        # 初始化网络基本参数
        logger.info("initializing network ....")
        self.layers = []
        for layerConf in self.globalConfig["layers"]:
            layer = globals()[layerConf["type"]](layerConf, self.globalConfig)
            self.layers.append(layer)
            logger.debug("Layer: " + layer.getLayerName())

        # layer setup，包括网络具体参数，如weight bias
        logger.info("setting up network ....")
        for ind in range(len(self.layers)):
            if ind == 0:
                self.layers[ind].setup(self.data, self.layers[ind])
            else:
                self.layers[ind].setup(self.layers[ind-1], self.layers[ind])

        # 优化
        logger.info("initialize solver [" + self.solverName + "]")
        self.solver = globals()[self.solverName](self)

    def optimize(self):
        logger.info("SGML optimize begin")
        self.solver.doOptimize()

