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
    def loss(X, y, W):
        pass

class LF_SVM(LossFunction):
    def loss(X, y, W):
        scores = W.dot(x)
        margins = np.maxium(0, scores - score[y] + 1)
        margins[y] = 0
        return np.sum(margins)

class LF_Softmax(LossFunction):
    def loss(X, y, W):
        scores = W.dot(x)
        sum = np.sum(np.exp(scores))
        s_yi = math.exp(scores[y])
        return - math.log(s_yi / sum)