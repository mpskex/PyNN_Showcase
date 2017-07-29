#coding:utf-8
import math
import random
import numpy as np

#   Neural Network Showcase
#   mpskex@github
#   2017

class DataGenerator(object):
    def __init__(self, cl=2, sz=50, upb=0, lwb = 10, dm=2):
        #   class in data
        self.cl = cl
        #   size of element of each class
        self.sz = sz
        #   dimension of data
        self.dm = dm
        #   upper bound of data
        self.upb = upb
        #   lower bound of data
        self.lwb = lwb
        #   initialize the data
        self.data = np.zeros((self.cl, self.sz, self.dm), np.float)
    def gen(self):
        pass

class DG_random(DataGenerator):
    def gen(self):
        for i in range(self.cl):
            for j in range(self.sz):
                for k in range(self.dm):
                    self.data[i][j][k] = random.uniform(self.upb, self.lwb)
        return self.data 

#   TO-DO make other DataGenerater