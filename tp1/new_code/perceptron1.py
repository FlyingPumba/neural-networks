import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import data as d
import datautil as du
import ocrletters as l
import random as rnd
from copy import deepcopy

class slPerceptron():
    
    def __init__(self):
        self.iNeurons = 26 # Includes threshold
        self.oNeurons = 5
        
    def train(self, origDataset, lr, epochs, epsilon, step = False, stochastic = False, maxError = False, batch = False):
        self.W = np.random.uniform(-0.1, 0.1, [self.iNeurons, self.oNeurons])
        dataset = deepcopy(origDataset)
        error = epsilon
        self.errors = []
        
        while epochs > 0 and (step or error >= epsilon):
            tempError = []
            
            if stochastic == True:
                np.random.shuffle(dataset)
            
            if batch == True:
                delta = np.zeros([self.iNeurons, self.oNeurons])
            
            for inp, dOut in dataset:
                
                #print np.array([inp]).shape
                #print self.W.shape
                res = np.dot(inp, self.W)
                #print res.shape
                
                if step == True:
                    diff = dOut - np.sign(res)
                else:
                    diff = dOut - np.tanh(res)
                
                tError = np.dot(diff, diff)
                tempError.append(tError)
                
                if batch == False:
                    #self.W += lr * np.dot(np.array([inp]).T, np.array([diff]))
                    self.W += lr * np.array([inp]).T * diff
                else:
                    delta += lr * np.array([inp]).T * diff
            
            if batch == True:
                self.W += delta
            
            if maxError == False:
                error = sum(tempError)
            else:
                error = max(tempError)
            self.errors.append(error)
            
            epochs -= 1
            
    def activate(self, input, step = False):
        if step == False:
            return np.tanh(np.dot(inp, self.W))
        else:
            return np.sign(np.dot(inp, self.W))
            
    def plotError(self):
        plt.plot(self.errors)
        plt.ylabel("network error")
        plt.xlabel("epoch number")
        plt.show()
            
if __name__ == "__main__":
    
    def addThreshold(X):
        Y = np.zeros(X.size + 1)
        for j in xrange(X.size):
            Y[j] = X[j]
        Y[X.size] = 1

        return Y
    
    trainingset = [
        (addThreshold(l.A), np.array([-1,-1,-1,-1, 1])), # A -> 1
        (addThreshold(l.B), np.array([-1,-1,-1, 1,-1])), # B -> 2
        (addThreshold(l.C), np.array([-1,-1,-1, 1, 1])), # C -> 3
        (addThreshold(l.D), np.array([-1,-1, 1,-1,-1])), # D -> 4
        (addThreshold(l.E), np.array([-1,-1, 1,-1, 1])), # E -> 5
        (addThreshold(l.F), np.array([-1,-1, 1, 1,-1])), # F -> 6
        (addThreshold(l.G), np.array([-1,-1, 1, 1, 1])), # G -> 7
        (addThreshold(l.H), np.array([-1, 1,-1,-1,-1])), # H -> 8
        (addThreshold(l.I), np.array([-1, 1,-1,-1, 1])), # I -> 9
        (addThreshold(l.J), np.array([-1, 1,-1, 1,-1])), # J -> 10
        (addThreshold(l.K), np.array([-1, 1,-1, 1, 1])), # K -> 11
        (addThreshold(l.L), np.array([-1, 1, 1,-1,-1])), # L -> 12
        (addThreshold(l.M), np.array([-1, 1, 1,-1, 1])), # M -> 13
        (addThreshold(l.N), np.array([-1, 1, 1, 1,-1])), # N -> 14
        (addThreshold(l.O), np.array([-1, 1, 1, 1, 1])), # O -> 15
        (addThreshold(l.P), np.array([ 1,-1,-1,-1,-1])), # P -> 16
        (addThreshold(l.Q), np.array([ 1,-1,-1,-1, 1])), # Q -> 17
        (addThreshold(l.R), np.array([ 1,-1,-1, 1,-1])), # R -> 18
        (addThreshold(l.S), np.array([ 1,-1,-1, 1, 1])), # S -> 19
        (addThreshold(l.T), np.array([ 1,-1, 1,-1,-1])), # T -> 20
        (addThreshold(l.U), np.array([ 1,-1, 1,-1, 1])), # U -> 21
        (addThreshold(l.V), np.array([ 1,-1, 1, 1,-1])), # V -> 22
        (addThreshold(l.W), np.array([ 1,-1, 1, 1, 1])), # W -> 23
        (addThreshold(l.X), np.array([ 1, 1,-1,-1,-1])), # X -> 24
        (addThreshold(l.Y), np.array([ 1, 1,-1,-1, 1])), # Y -> 25
        (addThreshold(l.Z), np.array([ 1, 1,-1, 1,-1])), # Z -> 26
        ]
    
    net = slPerceptron()
    
    net.train(trainingset, 0.01, 500, 0.01, step = False, stochastic = True, maxError = False, batch = False)
    
    for inp, dOut in trainingset:
        print dOut, ", ", net.activate(inp, step = False), ", ", dOut == net.activate(inp, step = True)
        
    net.plotError()
