import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data as d
import datautil as du
import ocrletters as l
import random as rnd

class HopfieldNetwork():
    """Memoria asociativa de Hopfield - Parte 1 - Ejercicio 2"""
    def __init__(self):
        self.cantNeuronas = 196

    def createWeights(self, memories):
        self.W = np.zeros((self.cantNeuronas, self.cantNeuronas))
        for X in memories:
            self.W += np.outer(X, X)
        self.W = self.W / self.cantNeuronas
        self.W = self.W - np.diag(np.diag(self.W))

    def energy(self, S, W):
        return -0.5 * np.dot(np.dot(S, W), S.T)

    def activate(self, X, synch=False):
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
                    S[i] = np.sign(np.dot(S, self.W[:,i]))

            E = self.energy(S,self.W)
            print "E: %s" % E
            Eh.append(E)
            # self.plotEnergy(Eh)
            # show(E,S)
        plt.ioff()
        return S

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

    def plotEnergy(self, energyHistory):
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
    memories = d.OCR.trainingset

    # ========== TRAINING ==========
    net.createWeights(memories)
    print "Weights: %s" % net.W

    # ========== VALIDATION ==========
    
    # test that for all the memories, the ouput of the activation is the same
    print "\n VALIDATION with original letters\n"
    for X in memories:
        output = net.activate(X)
        if (output == X).all():
            print "RIGHT memory"
        else:
            print "WRONG memory"

    # test some letters with noise
    print "\n VALIDATION with modified letters\n"
    # each letter has 196 bits so..
    Amod = []
    Amod.append(du.getTestSetWithNoise(l.A, 0.05, plot=True)) #letter A with 10 bits switched
    Amod.append(du.getTestSetWithNoise(l.A, 0.1, plot=True)) #letter A with 20 bits switched
    Amod.append(du.getTestSetWithNoise(l.A, 0.15, plot=True)) #letter A with 29 bits switched
    Amod.append(du.getTestSetWithNoise(l.A, 0.2, plot=True)) #letter A with 39 bits switched

    for X in Amod:
        output = net.activate(X)
        if (output == l.A).all():
            print "RIGHT letter A"
        else:
            print "WRONG letter A"

    Gmod = []
    Gmod.append(du.getTestSetWithNoise(l.G, 0.05)) #letter G with 10 bits switched
    Gmod.append(du.getTestSetWithNoise(l.G, 0.1)) #letter G with 20 bits switched
    Gmod.append(du.getTestSetWithNoise(l.G, 0.15)) #letter G with 29 bits switched
    Gmod.append(du.getTestSetWithNoise(l.G, 0.2)) #letter G with 39 bits switched

    for X in Gmod:
        output = net.activate(X)
        if (output == l.G).all():
            print "RIGHT letter G"
        else:
            print "WRONG letter G"

    Mmod = []
    Mmod.append(du.getTestSetWithNoise(l.M, 0.05)) #letter M with 10 bits switched
    Mmod.append(du.getTestSetWithNoise(l.M, 0.1)) #letter M with 20 bits switched
    Mmod.append(du.getTestSetWithNoise(l.M, 0.15)) #letter M with 29 bits switched
    Mmod.append(du.getTestSetWithNoise(l.M, 0.2)) #letter M with 39 bits switched

    for X in Mmod:
        output = net.activate(X)
        if (output == l.M).all():
            print "RIGHT letter M"
        else:
            print "WRONG letter M"

    Wmod = []
    Wmod.append(du.getTestSetWithNoise(l.W, 0.05)) #letter W with 10 bits switched
    Wmod.append(du.getTestSetWithNoise(l.W, 0.1)) #letter W with 20 bits switched
    Wmod.append(du.getTestSetWithNoise(l.W, 0.15)) #letter W with 29 bits switched
    Wmod.append(du.getTestSetWithNoise(l.W, 0.20)) #letter W with 39 bits switched

    for X in Wmod:
        output = net.activate(X)
        if (output == l.W).all():
            print "RIGHT letter W"
        else:
            print "WRONG letter W"