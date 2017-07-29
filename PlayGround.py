#coding:utf-8
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

class PlayGround(object):
    def __init__(self):
        self.generator = dg.DG_random(cl=3)
        self.data = self.generator.gen()
    def visualize(self):
        color = [1.0,0.0,0.0]
        if self.generator.dm==2:
            for n in range(self.generator.cl):
                for s in range(self.generator.sz):
                    plt.scatter(self.data[n][s][0], self.data[n][s][1], c=(color), alpha=0.5)
                color[0] = abs((color[0] - 0.3)%1)
                color[1] = abs((color[1] - 0.5)%1)
                color[2] = abs((color[2] - 0.7)%1)
            plt.show()
        else:
            print "Unable to visualize!"
if __name__ == '__main__':
    pg = PlayGround()
    pg.visualize()