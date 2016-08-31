# coding: utf-8
# system package
import sys, os
import json

# package
from src.util.UtilTool import *
from SGMLCore import SGMLNN


def LoadConfig():
    # 读取配置文件
    pwd = os.path.abspath('.')
    configFile = pwd + '\..\..\config\\test.json'
    f = file(configFile)
    jsonConfig = json.load(f)
    return jsonConfig

def initializeNeuronNetwork(jsonConfig):
    return SGMLNN(jsonConfig)

def main():
    nnJsonConfig = LoadConfig()
    logger.debug("Initialize neuronNetwork[" + nnJsonConfig["nn_name"] + "]")
    NN = initializeNeuronNetwork(nnJsonConfig)
    logger.debug("Done initialization for neuronNetwork[" + NN.neuronNetworkName + "]")
    NN.optimize()

if __name__=="__main__":
    main()






