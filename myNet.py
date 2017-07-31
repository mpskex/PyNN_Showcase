#coding: utf-8
import math
import numpy            as np
import Neurons          as nr
import Layers           as lyr
import LossFunctions    as lf
import DataGenerators   as dg
import LoadnSave        as LoS

#   Neural Network Showcase
#   mpskex@github
#   2017

class myNet(object):
    def __init__(self, fan_in, lr=0.01):
        #   learning rate
        self.lr = lr
        self.base_lr = lr
        self.fan_in = fan_in
        self.fan_out = 2
        self.loss = np.zeros(self.fan_out, np.float)
        self.__init_layers__()
        self.__init_loss_function__()
    def __init_layers__(self):
        #   5-10-4-2 hierachy
        #   This is design for my Porn detector 
        #   using pre-proc features to train this net
        #   stack layers
        self.layers = []
        #   layer-1
        self.layers.append(lyr.myFC(10, 5))
        #   layer-2
        self.layers.append(lyr.myFC(4, 10))
        #   output
        #   two class classifier
        self.layers.append(lyr.myFC(2, 4))
    def __init_loss_function__(self):
        self.LF = lf.LF_Softmax()
    def __update_learning_rate__(self):
        #   Constant learning rate
        self.lr = self.base_lr
    def forward(self, X):
        #   layer-1 forward
        X1 = self.layers[0].forward(X)
        #   layer-2 forward
        X2 = self.layers[1].forward(X1)
        #   layer-3 forward
        #   output layer
        self.out = self.layers[2].forward(X2)
        return self.out
    def backward(self, Loss, label):
        dG2 = self.layers[2].backward(Loss, self.lr)
        dG1 = self.layers[1].backward(dG2, self.lr)
        dG0 = self.layers[0].backward(dG1, self.lr)
    def calc_loss(self, label):
        return self.LF.loss(self.out, label)
    def epoch(self, dataset, labelset):
        #   X is a dozen of training data
        #   when the net go over a whole set of data
        #   that means an epoch training is done
        for i in range(len(dataset)):
            self.forward(dataset[i])
            label = np.ones(self.fan_out)
            label[labelset[i]] = 1
            dS = self.out - label
            self.loss = self.calc_loss(labelset[i])
            self.backward(dS, labelset[i])
        print "Loss is ", self.loss
        print "dScore is ", dS, " label is ", labelset[i]
        self.__update_learning_rate__()
    def train(self, epoch, dataset, labelset):
        for i in range(epoch):
            print "Epoch ", i
            self.epoch(dataset, labelset)
    def show_weights(self):
        for i in self.layers:
            i.show_weights()

if __name__ == '__main__':
    '''
    net = myNet()
    print net.forward(np.array([1,2,3,4,5]))
    print net.backward(np.array([3]))
    '''
    label, VecList = LoS.LoadMat("prob.mat")
    net = myNet(5, lr=0.1)
    #net.show_weights()
    #net.epoch(VecList, label)
    #net.show_weights()
    net.train(10, VecList, label)