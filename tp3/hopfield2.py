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

    def activate(self, X, synch=False, plotEnergy=False, plotOutput=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()

        if plotOutput:
            plt.clf()
            letterReshaped = np.copy(S).reshape(14,14)
            plt.imshow(letterReshaped, interpolation='none', cmap=cm.gray)
            plt.draw()

        while(not np.array_equal(S,Saux)):
            Saux = np.copy(S)

            if synch:
                S = np.sign(np.dot(S,self.W))
            else:
                I = np.random.permutation(self.cantNeuronas)
                for i in I:
                    #print "I: %s" % S
                    S[i] = np.sign(np.dot(S, self.W[:,i]))

            if plotOutput:
                plt.clf()
                letterReshaped = np.copy(S).reshape(14,14)
                plt.imshow(letterReshaped, interpolation='none', cmap=cm.gray)
                plt.draw()

            E = self.energy(S,self.W)
            #print "E: %s" % E
            Eh.append(E)
            if plotEnergy:
                self.plotEnergy(Eh)
        if plotEnergy or plotOutput:
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

    def getEspuriousRate(self, memories, pattern, noise, times):
        count = 0.0
        for i in xrange(1, times):
            X = du.getPatternWithNoise(pattern, noise)
            #du.plotLetter(X, saveFile=True)
            result = self.activate(np.copy(X))
            if (pattern != result).any():
                count = count + 1.0
        print "Espurious rate of %.1f %%  with %.1f %%  noise " % (count / times * 100.0, noise * 100)


# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    #plt.ion()
    net = HopfieldNetwork()

    # generate the memories
    #memories = d.OCR.trainingset
    memories = np.asarray([l.A, l.G, l.M, l.W])

    # ========== TRAINING ==========
    net.createWeights(memories)
    print "Weights: %s" % net.W

    # ========== VALIDATION ==========
    
    # test that for all the memories, the ouput of the activation is the same
    print "\n VALIDATION with original letters\n"
    for X in memories:
        #du.plotLetter(X, saveFile=True)
        output = net.activate(np.copy(X))
        if (output == X).all():
            print "RIGHT memory"
        else:
            print "WRONG memory"

    # test some letters with noise
    print "\n VALIDATION with modified letters\n"

    print "\n First memory"
    net.getEspuriousRate(memories, memories[0], 0.05, 100)
    net.getEspuriousRate(memories, memories[0], 0.1, 1000)
    net.getEspuriousRate(memories, memories[0], 0.15, 1000)
    net.getEspuriousRate(memories, memories[0], 0.2, 1000)
    net.getEspuriousRate(memories, memories[0], 0.3, 1000)
    net.getEspuriousRate(memories, memories[0], 0.4, 1000)
    net.getEspuriousRate(memories, memories[0], 0.5, 1000)

    print "\n Second memory"
    net.getEspuriousRate(memories, memories[1], 0.05, 1000)
    net.getEspuriousRate(memories, memories[1], 0.1, 1000)
    net.getEspuriousRate(memories, memories[1], 0.15, 1000)
    net.getEspuriousRate(memories, memories[1], 0.2, 1000)
    net.getEspuriousRate(memories, memories[1], 0.3, 1000)
    net.getEspuriousRate(memories, memories[1], 0.4, 1000)
    net.getEspuriousRate(memories, memories[1], 0.5, 1000)

    print "\n Third memory"
    net.getEspuriousRate(memories, memories[2], 0.05, 1000)
    net.getEspuriousRate(memories, memories[2], 0.1, 1000)
    net.getEspuriousRate(memories, memories[2], 0.15, 1000)
    net.getEspuriousRate(memories, memories[2], 0.2, 1000)
    net.getEspuriousRate(memories, memories[2], 0.3, 1000)
    net.getEspuriousRate(memories, memories[2], 0.4, 1000)
    net.getEspuriousRate(memories, memories[2], 0.5, 1000)

    print "\n Fourth memory"
    net.getEspuriousRate(memories, memories[2], 0.05, 1000)
    net.getEspuriousRate(memories, memories[2], 0.1, 1000)
    net.getEspuriousRate(memories, memories[2], 0.15, 1000)
    net.getEspuriousRate(memories, memories[2], 0.2, 1000)
    net.getEspuriousRate(memories, memories[2], 0.3, 1000)
    net.getEspuriousRate(memories, memories[2], 0.4, 1000)
    net.getEspuriousRate(memories, memories[2], 0.5, 1000)