#coding: utf-8
import math
import numpy as np

#   Neural Network Showcase
#   mpskex@github
#   2017

#   There are lots of activation functions
#   The activation function get value from neurons
#   They decide whether the neurons are activated or not
#   tanh(x) and ReLU activation function are common
#   Also we give sigmoid and others for choices

class ActivationFunction(object):
    def __init__(self):
        pass
    def forward(self, x):
        #   As a part of the computable graph
        #   it should also be capable to
        #   compute both forward and backward
        pass
    def backward(self, dz):
        pass

class AF_tanh(ActivationFunction):
    def forward(self, x):
        #   tanh function is known as a good and symmetric function 
        #   But it dies when countering bigger or smaller
        #   stimulus, which is known as the gradient vanishing problem
        #   A good way to overcome it is to normalize all the input in
        #   shape, and it would perform well maybe
        self.x = x
        return math.tanh(x)
    def backward(self, dz):
        #   Caculating the local gradient
        return dz * (1 - math.tanh(self.x)**2)

class AF_ReLU(ActivationFunction):
    def forward(self, x):
        #   ReLU is good at handling in positive values
        #   And it's not saturate when x>0
        #   But you have to concern when x<=0
        self.x = x
        return max(0, x)
    def backward(self, dz):
        if self.x>0:
            return 1
        else:
            return 0

class AF_leakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        #   Attention: alpha here need to be in range [0,1)
        #   Upper bound below 0.1 is recommended
        self.alpha = alpha
    def forward(self, x, alpha=0.01):
        #   leaky-ReLU is good for it can deal with smaller x
        #   and it have a hyperparameter which need tuning
        self.x = x
        return max(alpha*x, x)
    def backward(self, dz):
        if self.x>0:
            return 1
        else:
            return self.alpha

#   TO-DO: sigmoid & ELU