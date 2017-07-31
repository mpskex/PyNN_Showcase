#coding: utf-8
import math
import numpy    as np
import Neurons  as nr

#   Neural Network Showcase
#   mpskex@github
#   2017

#   X, Y, W are vectors in forward function
#   dX, dG are vectors in backward propogation
#   We define here a Layer object

class FullyConnectedLayers(object):
    def __init__(self, size, fan_in):
        self.size = size
        self.fan_in = fan_in
        self.neurons = []
        self.__init_neurons__()
    def __init_neurons__(self):
        for i in range(self.size):
            self.neurons.append(nr.Neuron(fan_in))
    def forward(self, X):
        #   We collect the whole layer output as a vector
        out = np.zeros(self.size, np.float)
        for i in range(self.size):
            out[i] = self.neurons[i].forward(X)
        return out
    def backward(self, dG):
        dX = np.zeros(self.fan_in, np.float)
        for i in range(self.size):
            dX += self.neurons[i].backward(dG[i])
        return dX
    def show_weights(self):
        print "layer weights:"
        for i in range(self.size):
            print self.neurons[i].W
        
if __name__ == '__main__':
    FC = FullyConnectedLayers(2,2)
    FC.show_weights()
    print "forward:"
    print FC.forward(np.array([1, 1]))
    print "backward:"
    print FC.backward(np.array([0.1, 0.1]))