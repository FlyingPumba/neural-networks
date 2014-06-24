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

    def activate(self, X, sync=False, plotEnergy=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()

        while(not np.array_equal(S,Saux)):
            Saux = np.copy(S)

            if sync:
                S = np.sign(np.dot(S,self.W))
            else:
                I = np.random.permutation(self.cantNeuronas)
                for i in I:
                    #print "I: %s" % S
                    S[i] = np.sign(np.dot(S, self.W[:,i]))

            E = self.energy(S,self.W)
            #print "E: %s" % E
            Eh.append(E)
            if plotEnergy and len(Eh) > 1:
                self.plotEnergy(Eh)
        if plotEnergy and len(Eh) > 1:
            self.plotEnergy(Eh, saveFile=True)
        plt.ioff()
        return S

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

    def plotEnergy(self, energyHistory, saveFile=False):
        plt.clf()
        plt.xlabel('Iteracion')
        plt.ylabel('Energia')
        plt.plot(energyHistory)
        if saveFile:
            filekey = np.random.randint(1000)
            fileName = 'hopfield1-energy-%d.png' % filekey
            print fileName
            figure = plt.gcf() # get current figure
            figure.set_size_inches(10, 8) #this will give us a 800x600 image
            # when saving, specify the DPI
            plt.savefig(fileName, bbox_inches='tight', dpi = 100)
        plt.draw()

    def isEspurious(self, memories, pattern, note="", doPrint=True):
        output = self.activate(np.copy(pattern))
        if any((output == x).all() for x in memories):
            if (doPrint):
                print "Pattern %s ended in a known memory." % pattern + " " + note 
            return False
        else:   
            if (doPrint):
                print "Pattern %s ended in a not known memory. FOUND ESPURIOS state" % pattern + " " + note 
            return True

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    #plt.ion()
    net = HopfieldNetwork()
    orthNet = HopfieldNetwork()
    # generate the memories
    memories = []
    memories.append([1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    memories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    memories.append([1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1])
    print "Memories: %s" % memories

    orthMemories = []
    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1])
    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1])

    net.createWeights(memories)
    orthNet.createWeights(orthMemories)
    #print "Weights: %s" % net.W

    # ========== VALIDATION with original memories ==========
    print "\n VALIDATION with original memories\n"
    for X in memories:
        output = net.activate(np.copy(X), plotEnergy=True)
        if (output == X).all():
            print "Memory %s is RIGHT (asyncronic activation)" % X
        else:
            print "Memory %s is WRONG (asyncronic activation)" % X

    print "\n VALIDATION with original memories (syncronic activation)\n"
    for X in memories:
        output = net.activate(np.copy(X), plotEnergy=True, sync=True)
        if (output == X).all():
            print "Memory %s is RIGHT (syncronic activation)" % X
        else:
            print "Memory %s is WRONG (syncronic activation)" % X

    
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
        net.isEspurious(memories, X, "(distorted memory 0)")

    memory1Set = []
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.05))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.1))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.15))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.2))
    memory1Set.append(du.getPatternWithNoise(memories[1], 0.3))

    for X in memory1Set:
        net.isEspurious(memories, X, "(distorted memory 1)")

    memory2Set = []
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.05))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.1))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.15))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.2))
    memory2Set.append(du.getPatternWithNoise(memories[2], 0.3))

    for X in memory2Set:
        net.isEspurious(memories, X, "(distorted memory 2)")

    # ========== VALIDATION analytic espurious states ==========
    print "\n VALIDATION analytic espurious states\n"
    espurious = []
    espurious.append([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1])
    espurious.append([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    espurious.append([-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1])


    for X in espurious:
        net.isEspurious(memories, X, "(inverted memory = espurious)")

    # ========== VALIDATION empiric espurious states ==========
    print "\n VALIDATION empiric espurious states\n"

    # get 1000 numbers within 1 and 2**20 (1048576)
    numbers = np.random.randint(1, 2**20, size=10000)
    #numbers = xrange(0,2**10)
    # transform them in binary
    numbers = [np.binary_repr(x, width=20) for x in numbers]

    offset = 1
    for i in xrange(len(numbers)):
        aux = numbers[i]
        lista = []
        # split binary representation into list
        lista[offset:offset+len(aux)] = list(aux)
        # convert list to numpy array
        lista = np.array(lista)
        # convert string representation to integers
        lista = lista.astype(np.int)
        # replace zeros with -1s
        lista[lista == 0] = -1
        # save the pattern
        numbers[i] = lista

    count = 0
    for X in numbers:
        if net.isEspurious(memories, X, "", False):
            count = count + 1

    count2 = 0
    for X in numbers:
        if orthNet.isEspurious(orthMemories, X, "", False):
            count2 = count2 + 1

    print "FOUND %d espirious states of %d with orthogonal memories" % (count, len(numbers))
    print "FOUND %d espirious states of %d with non orthogonal memories" % (count2, len(numbers))

