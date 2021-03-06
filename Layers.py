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

#   Fully Connected Layer
class FullyConnectedLayer(object):
    def __init__(self, size, fan_in):
        self.size = size
        self.fan_in = fan_in
        self.neurons = []
        self.__init_neurons__()
    def __init_neurons__(self):
        #   We can custom our own neurons
        for i in range(self.size):
            self.neurons.append(nr.Neuron(self.fan_in))
    def forward(self, X):
        #   We collect the whole layer output as a vector
        out = np.zeros(self.size, np.float)
        for i in range(self.size):
            out[i] = self.neurons[i].forward(X)
        return out
    def backward(self, dG, lr):
        dX = np.zeros(self.fan_in, np.float)
        for i in range(self.size):
            dX += self.neurons[i].backward(dG[i], lr)
        return dX
    def show_weights(self):
        for i in range(self.size):
            print self.neurons[i].W

class myFC1(FullyConnectedLayer):
    def __init_neurons__(self):
        for i in range(self.size):
            self.neurons.append(nr.myNr1(self.fan_in))

class myFC2(FullyConnectedLayer):
    def __init_neurons__(self):
        for i in range(self.size):
            self.neurons.append(nr.myNr2(self.fan_in))

#   TO-DO
#   Convolutional Layer
class ConvLayer(object):
    def __init__(self):
        pass
    def forward(self, X):
        pass
    def backward(self, dG):
        pass
    def show_kernel(self):
        pass
        


if __name__ == '__main__':
    FC = FullyConnectedLayer(2,2)
    FC.show_weights()
    print "forward:"
    print FC.forward(np.array([1, 1]))
    print "backward:"
    print FC.backward(np.array([0.1, 0.1]),0.1)