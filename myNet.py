#coding: utf-8
import math

import numpy            as np
import Neurons          as nr
import Layers           as lyr
import LossFunctions    as lf
import DataGenerators   as dg

#   Neural Network Showcase
#   mpskex@github
#   2017

class myNet(object):
    def __init__(self, lr=0.01):
        #   learning rate
        self.base_lr = lr
        self.__init_layers__()
    def __init_layers__(self):
        #   5-10-4-1 hierachy
        #   This is design for my Porn detector 
        #   using pre-proc features to train this net
        #   stack layers
        self.layers = []
        #   layer-1
        self.layers.append(lyr.myFC(10, 5))
        #   layer-2
        self.layers.append(lyr.myFC(4, 10))
        #   output
        self.layers.append(lyr.myFC(1, 4))
    def __update_learning_rate__(self):
        pass
    def forward(self, X):
        #   layer-1 forward
        X1 = self.layers[0].forward(X)
        #   layer-2 forward
        X2 = self.layers[1].forward(X1)
        #   layer-3 forward
        #   output layer
        return self.layers[2].forward(X2)
    def backward(self, Loss):
        dG2 = self.layers[2].backward(Loss)
        dG1 = self.layers[1].backward(dG2)
        dG0 = self.layers[0].backward(dG1)

if __name__ == '__main__':
    net = myNet()