#coding: utf-8
import math

import numpy            as np
import Neurons          as nr
import LossFunctions    as lf
import DataGenerators   as dg

import matplotlib
import matplotlib.pyplot    as plt

#   Neural Network Showcase
#   mpskex@github
#   2017

class myNet(object):
    def __init__(self):
        self.nodes = []
        
