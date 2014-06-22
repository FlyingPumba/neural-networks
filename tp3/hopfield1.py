import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datautil as du
import random as rnd

class HopfieldNetwork():
    """Memoria asociativa de Hopfield - Parte 1 - Ejercicio 1"""
    def __init__(self):
        self.cantNeuronas = 20

    def createWeights(self, memories):
        self.W = np.zeros((self.cantNeuronas, self.cantNeuronas))
        for X in memories:
            self.W += np.outer(X, X)
        self.W = self.W / self.cantNeuronas
        self.W = self.W - np.diag(np.diag(self.W))

    def energy(self, S, W):
        return -0.5 * np.dot(np.dot(S, W), S.T)

    def activate(self, X, synch=False, plotEnergy=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()

        while(not np.array_equal(S,Saux)):
            Saux = np.copy(S)

            if synch:
                S = np.sign(np.dot(S,self.W))
            else:
                I = np.random.permutation(self.cantNeuronas)
                for i in I:
                    #print "I: %s" % S
                    S[i] = np.sign(np.dot(S, self.W[:,i]))

            E = self.energy(S,self.W)
            print "E: %s" % E
            Eh.append(E)
            if plotEnergy:
                self.plotEnergy(Eh)
        if plotEnergy:
            raw_input()
        plt.ioff()
        return S

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
    #plt.ion()
    net = HopfieldNetwork()

    # generate the memories
    memories = []
    cantMemorias = 3

    memories.append([1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    memories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    memories.append([1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1])

    print "Memories: %s" % memories
    net.createWeights(memories)
    #print "Weights: %s" % net.W

    # ========== VALIDATION with original memories ==========
    print "\n VALIDATION with original memories\n"
    for X in memories:
        output = net.activate(np.copy(X))
        if (output == X).all():
            print "RIGHT memory"
        else:
            print "WRONG memory"
    
    # ========== VALIDATION with modified memories ==========
    print "\n VALIDATION with modified memories\n"
    memory0Set = []
    # each memory has 20 bits so..
    memory0Set.append(du.getPatternWithNoise(memories[0], 0.05)) #memory with 1 bit switched
    memory0Set.append(du.getPatternWithNoise(memories[0], 0.1)) #memory with 2 bits switched
    memory0Set.append(du.getPatternWithNoise(memories[0], 0.15)) #memory with 3 bits switched
    memory0Set.append(du.getPatternWithNoise(memories[0], 0.2)) #memory with 4 bits switched
    memory0Set.append(du.getPatternWithNoise(memories[0], 0.3)) #memory with 6 bits switched

    for X in memory0Set:
        output = net.activate(np.copy(X))
        if (output == memories[0]).all():
            print "RIGHT memory 0"
        else:
            print "WRONG memory 0"

    memory1Set = []
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.05))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.1))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.15))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.2))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.3))

    for X in memory1Set:
        output = net.activate(np.copy(X))
        if (output == memories[1]).all():
            print "RIGHT memory 1"
        else:
            print "WRONG memory 1"

    memory2Set = []
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.05))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.1))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.15))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.2))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.3))

    for X in memory2Set:
        output = net.activate(np.copy(X))
        if (output == memories[2]).all():
            print "RIGHT memory 2"
        else:
            print "WRONG memory 2"