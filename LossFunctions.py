#coding: utf-8
import math
import numpy as np

#   Neural Network Showcase
#   mpskex@github
#   2017

#   Loss Functions
#   SVM and Softmax is common in loss functions

#   Here the X, Y, W are vectors
#   We use numpy to deal with vectors and matrix
#   X is the input vector to caculate the score
#   W is the input weight
#   y is the label
#   the score vector is caculated in equation score = W * Xj

class LossFunction(object):
    def loss(self, scores, label):
        pass

class LF_Hinge(LossFunction):
    def loss(self, scores, label):
        margins = np.maximum(0, scores - scores[label] + 1)
        margins[label] = 0
        return np.sum(margins)

class LF_Softmax(LossFunction):
    def loss(self, scores, label):
        sum = np.sum(np.exp(scores))
        s_yi = math.exp(scores[label])
        return - math.log(s_yi / sum)