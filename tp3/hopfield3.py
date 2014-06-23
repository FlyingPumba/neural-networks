import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data as d
import datautil as du
import ocrletters as l
import random as rnd

class HopfieldNetwork():
    """Memoria asociativa de Hopfield - Parte 2 - Ejercicio 3"""
    def __init__(self):
        self.cantNeuronas = 100

    def createWeights(self, memories):
        self.W = np.zeros((self.cantNeuronas, self.cantNeuronas))
        for X in memories:
            self.W += np.outer(X, X)
        self.W = self.W / self.cantNeuronas
        self.W = self.W - np.diag(np.diag(self.W))

    def energy(self, S, W):
        return -0.5 * np.dot(np.dot(S, W), S.T)
        
    def sigmoid(self, x, temp) :
        return 1/(1 + np.exp( -x / temp ))

    def activate(self, X, temp, dist, hamming = False, plotEnergy=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()

        currentDist = self.dist(S, Saux, hamming)
        print currentDist

        while currentDist > dist:
            Saux = np.copy(S)
            I = np.random.permutation(self.cantNeuronas)
            for i in I:
                S[i] = np.sign(self.sigmoid(np.dot(S, self.W[:,i]), temp) - np.random.uniform(0,1))

            currentDist = self.dist(S, Saux, hamming)
            print currentDist

            E = self.energy(S,self.W)
            #print "E: %s" % E
            Eh.append(E)

            if plotEnergy:
                self.plotEnergy(Eh)

        if plotEnergy:
            raw_input()

        plt.ioff()
        return S

    def dist(self, a, b, hamming=False):
        if hamming:
            return self.distHamming(a, b)
        else:
            return self.distEuclidean(a, b)

    def distEuclidean(self, a, b):
        return np.sqrt(np.sum((b - a)**2))
   
    def distHamming(self, a, b):
        dist = 0
        for i in xrange(len(a)) :
            if a[i] != b[i]:
                dist += 1
        return dist

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

    def plotEnergy(self, energyHistory):
        plt.clf()
        plt.xlabel('Iteracion')
        plt.ylabel('Energia')
        plt.plot(energyHistory)
        plt.draw()

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    net = HopfieldNetwork()

    # temperature
    temp = 0.4

    # threshold distance to consider one pattern the same as other
    tDist = 0.01

    # generate the memories
    cantMemorias = 10
    memories = []
 
    while cantMemorias > 0 :
        memories.append(np.round(np.random.sample(100)) * 2 - 1)
        cantMemorias -= 1

    net.createWeights(memories)


    # ========== VALIDATION ==========

    # test that for all the memories, the ouput of the activation is the same
    print "\n VALIDATION with original memories\n"
    for X in memories:
        output = net.activate(np.copy(X), temp, tDist,  plotEnergy=True)
        if (output == X).all():
            print "RIGHT memory"
        else:
            print "WRONG memory"
