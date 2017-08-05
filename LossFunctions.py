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
    def __init__(self):
        pass
    def forward(self, scores, label):
        pass

class LF_Hinge(LossFunction):
    def forward(self, scores, label):
        loss = 0
        for i in range(len(scores)):
            if i != label:
                loss += max(0, 1 + scores[i])
            else:
                loss += max(0, 1 - scores[i])
        return loss

class LF_Softmax(LossFunction):
    def forward(self, scores, label):
        sum = np.sum(np.exp(scores))
        s_yi = math.exp(scores[label])
        return - math.log(s_yi / sum)

if __name__ == '__main__':
    hinge = LF_Hinge()
    softmax = LF_Softmax()
    print "hinge loss :", hinge.forward(np.array([0.121, -0.023]), 1)
    print "softmax loss :", softmax.forward(np.array([0.121, -0.023]), 1)