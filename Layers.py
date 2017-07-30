#coding: utf-8
import math
import numpy    as np
import Neurons  as nr

#   Neural Network Showcase
#   mpskex@github
#   2017

class FullyConnectedLayers(object):
    def __init__(self, size, fan_in):
        self.size = size
        self.fan_in = fan_in
        self.neurons = []
        for i in range(size):
            self.neurons.append(nr.Neuron(fan_in))
        