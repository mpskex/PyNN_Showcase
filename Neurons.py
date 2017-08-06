#coding: utf-8
import numpy as np
import ActivationFunctions
import UpdateFunctions

import LossFunctions    as lf

#   Neural Network Showcase
#   mpskex@github
#   2017

#   Actually Neural Network is a Computable Graph
#   with high dimensional correlationship as a system

#   Here introduced a simple structure of neurons
#   Let's see how these compute unit works

class Neuron(object):
    def __init__(self, fan_in):
        #   Initializing the weights
        self.W = 0.2 * np.random.rand(fan_in)
        #   setting bias for neuron
        #   default is zero
        self.b = 0
        #   create activation function
        self.__init_ActivationFunction__()
        #   create update function
        self.__init_UpdateFunction__()
    def __init_ActivationFunction__(self):
        #   Here we can redefine by redeclare different activation functions
        self.AF = ActivationFunctions.AF_ReLU()
    def __init_UpdateFunction__(self):
        #   We define a update function for weight update
        self.UF = UpdateFunctions.UF_SGD()
    def forward(self, X):
        #   Do the forward computing
        self.X = X
        self.Y = np.dot(self.W, self.X.T) + self.b
        return self.AF.forward(self.Y)
    def backward(self, dG, lr):
        #   Do the Back-Prop computing
        #   Using chain rule
        #   for each dG/dXi
        #   the local gradient is 
        #   W it self times G
        out = self.AF.backward(dG) / self.W
        self.W += self.UF.update(out, lr)
        return out

class myNr1(Neuron):
    def __init_ActivationFunction__(self):
        self.AF = ActivationFunctions.AF_tanh()
    def __init_UpdateFunction__(self):
        self.UF = UpdateFunctions.UF_SGD()

class myNr2(Neuron):
    def __init_ActivationFunction__(self):
        self.AF = ActivationFunctions.AF_tanh()
    def __init_UpdateFunction__(self):
        self.UF = UpdateFunctions.UF_Momentum()

if __name__ == '__main__':
    n = Neuron(2)
    print "n's weight" , n.W
    print n.forward(np.array([1, -2]))
    print n.backward(0.11, 0.01)
    print "n's weight" , n.W