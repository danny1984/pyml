# coding = utf-8

from LayerBase import LayerBase

class Conv(LayerBase):

    def __init__(self, config, globalConfig):
        super(Conv, self).__init__(config, globalConfig)

    def setup(self, bottom, top):
        super(Conv, self).setup(bottom, top)

    def forward(self, preLayer):
        super(Conv, self).forward()

    def backward(self, postLayer):
        super(Conv, self).backward()