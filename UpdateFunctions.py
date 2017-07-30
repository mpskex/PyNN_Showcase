#coding: utf-8
import math
import numpy as np

#   Neural Network Showcase
#   mpskex@github
#   2017

#   Update Function is introduced to update the parameters
#   And here we define functions returns the update value
#   instead of the whole thing at a time, so please pay attention to it

#   learning rate is named as lr
#   gradient of x is named as dx

class UpdateFunction(object):
    def update(self, dx, lr):
        pass

class UF_SGD(UpdateFunction):
    #   Simple Gradient Descent is working
    #   But with poor speed to find the lowest point
    #   SGD will have slow progress along flat direction
    #   jitter along steep one
    #   It might wiggle and zig-zag a bit
    def update(self, dx, lr):
        return - lr * dx

class UF_Momentum(UpdateFunction):
    #   Momentum is physically describe a problem 
    #   like a ball roll down in a loss function with friction
    def __init__(self, mu=0.9):
        self.mu = mu
        self.v = 0
    def update(self, dx, lr):
        self.v = self.mu * self.v - lr * dx 
        return self.v

class UF_NAG(UpdateFunction):
    #   TO-DO
    #   description
    def __init__(self, mu=0.9):
        self.mu = mu
        self.v = 0
        self.v_prev = 0
    def update(self, dx, lr):
        self.v_prev = v
        self.v = self.mu * self.v - lr * dx
        return -self.mu * self.v_prev + (1 + self.mu) * self.v

class UF_AdaGrad(UpdateFunction):
    #   Added element-wise scaling of the gradient 
    #   based on the historical sum of square in each dimension
    def __init__(self):
        self.cache = 0
    def update(self, dx, lr):
        self.cache += dx**2
        return - lr * dx / (math.sqrt(self.cache) + 1e-7)

class UF_RMSProp(UpdateFunction):
    #   It provide a decay parameter to ignore the effect 
    #   which brings from very early update
    #   Decay rate is define as dr
    def __init__(self, dr=0.99):
        self.cache = 0
        self.dr = dr
    def update(self, dx, lr):
        self.cache = self.dr * self.cache + (1 - self.dr) * dx**2
        return - lr * dx / (math.sqrt(self.cache) + 1e-7)

class UF_Adam(UpdateFunction):
    #   Mixed with momentum and RMSProp
    def __init__(self, b1=0.9, b2=0.995):
        self.b1 = b1
        self.b2 = b2
        self.m = 0
        self.v = 0
    def update(self, dx, lr):
        self.m = self.b1 * self.m + (1 - self.b1) * dx
        self.v = self.b2 * self.v + (1 - self.b2) * (dx**2)
        return - lr * self.m / (math.sqrt(self.v) + 1e-7)