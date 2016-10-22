# coding: utf-8
# system package
import sys, os
import json

# package
from src.util.UtilTool import *
from SGMLNN import SGMLNN


def LoadConfig():
    # 读取配置文件
    pwd = os.path.abspath('.')
    configFile = pwd + "\..\..\config\ShallowNN.json"
    #configFile = pwd + "\..\..\config\LeNetTest.json"
    logger.info("SGML starts, and going to read: " + configFile)
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






