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

    def activate(self, X, temp, synch=False, plotEnergy=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()

        while(not np.array_equal(S,Saux)):
            Saux = np.copy(S)

            if synch:
                S = np.sign(S*self.W)
            else:
                I = np.random.permutation(self.cantNeuronas)
                for i in I:
                    #print "I: %s" % S
                    #S[i] = np.sign(np.dot(S, self.W[:,i]))
                    S[i] = np.sign(self.sigmoid(np.dot(S, self.W[:,i]), temp) - np.random.uniform(0,1))

            E = self.energy(S,self.W)
            print "E: %s" % E
            Eh.append(E)
            if plotEnergy:
                self.plotEnergy(Eh)
        if plotEnergy:
            raw_input()
        plt.ioff()
        return S

    def activatewd(self, X, temp, desiredMem, dist, hamming = False, synch=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()
       
        if hamming = False :
            while(np.sqrt(numpy.sum((desiredMem - Saux)**2)) > dist) :
               
                if synch:
                    S = np.sign(S*self.W)
                else:
                    I = np.random.permutation(self.cantNeuronas)
                    for i in I:
                        #print "I: %s" % S
                        #S[i] = np.sign(np.dot(S, self.W[:,i]))
                        S[i] = np.sign(self.sigmoid(np.dot(S, self.W[:,i]), temp) - np.random.uniform(0,1))
 
                E = self.energy(S,self.W)
                #print "E: %s" % E
                Eh.append(E)
                # self.plotEnergy(Eh)
                # show(E,S)
                Saux = np.copy(S)
            return S
        else :
            while(distHamming(desiredMem, Saux) > dist) :
 
                if synch:
                    S = np.sign(S*self.W)
                else:
                    I = np.random.permutation(self.cantNeuronas)
                    for i in I:
                        #print "I: %s" % S
                        #S[i] = np.sign(np.dot(S, self.W[:,i]))
                        S[i] = np.sign(self.sigmoid(np.dot(S, self.W[:,i]), temp) - np.random.uniform(0,1))
 
                E = self.energy(S,self.W)
                #print "E: %s" % E
                Eh.append(E)
                # self.plotEnergy(Eh)
                # show(E,S)
                Saux = np.copy(S)
            return S
   
    def distHamming(a, b) :
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

    # generate the memories
    cantMemorias = 10
    memories = []
 
    while cantMemorias > 0 :
        memories.append(np.round(np.random.sample(100)) * 2 - 1)
        cantMemorias -= 1

    # sets the temperature
    temp = 0.4