#coding: utf-8

from src.util.UtilTool import *

class LayerBase(object):

    def __init__(self, config, globalConfig):
        logger.debug("Initialize layer(" + config["layer_name"] + ")" )
        self.name       = config["layer_name"]
        self.config     = config
        self.globalConfig = globalConfig

    def setup(self, bottom, top):
        logger.debug("Layer(" + self.getLayerName() + ") setting up ....")

    def forward(self):
        logger.debug("Layer(" + self.getLayerName() + ") forwarding ....")

    def backward(self):
        logger.debug("Layer(" + self.getLayerName() + ") backwarding ....")

    def getLayerName(self):
        return self.name

    def miniBatchPrepare(self):
        logger.debug("Layer(" + self.getLayerName() + ") miniBatchPrepare....")