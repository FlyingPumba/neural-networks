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
                print "Pattern %s ended in a not known memory. FOUND ESPURIOUS state" % pattern + " " + note 
            return True

    def getEspuriousRate(self, memories, pattern, noise, times):
    	count = 0.0
    	for i in xrange(1, times): 
    		X = du.getPatternWithNoise(pattern, noise)
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
    orthNet = HopfieldNetwork()
    # generate the memories
    memories = []
    memories.append([1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    memories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    memories.append([1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1])
    print "Memories: %s" % memories

    orthMemories = []
    #orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1])
    #orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1])

    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1])
    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1])
    orthMemories.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1])

    net.createWeights(memories)
    orthNet.createWeights(orthMemories)
    #print "Weights: %s" % net.W

    # ========== VALIDATION with original memories ==========
    print "\n VALIDATION with original memories\n"
    for X in memories:
        output = net.activate(np.copy(X))
        if (output == X).all():
            print "Memory %s is RIGHT (asyncronic activation)" % X
        else:
            print "Memory %s is WRONG (asyncronic activation)" % X

    print "\n VALIDATION with original memories (syncronic activation)\n"
    for X in memories:
        output = net.activate(np.copy(X), sync=True)
        if (output == X).all():
            print "Memory %s is RIGHT (syncronic activation)" % X
        else:
            print "Memory %s is WRONG (syncronic activation)" % X

    
    print "\n VALIDATION with original memories (no orthogonal network)\n"
    for X in orthMemories:
        output = orthNet.activate(np.copy(X))
        if (output == X).all():
            print "Memory %s is RIGHT (asyncronic activation)" % X
        else:
            print "Memory %s is WRONG (asyncronic activation)" % X

    print "\n VALIDATION with original memories (syncronic activation) (no orthogonal network)\n"
    for X in orthMemories:
        output = orthNet.activate(np.copy(X), sync=True)
        if (output == X).all():
            print "Memory %s is RIGHT (syncronic activation)" % X
        else:
            print "Memory %s is WRONG (syncronic activation)" % X

    # ========== VALIDATION with modified memories ==========
    print "\n VALIDATION with modified memories for the orthogonal network"

    print "\n First memory"
    net.getEspuriousRate(memories, memories[0], 0.05, 1000)
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

    print "\n VALIDATION with modified memories for the non orthogonal network"

    print "\n First memory"
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.05, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.1, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.15, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.2, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.3, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.4, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[0], 0.5, 1000)

    print "\n Second memory"
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.05, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.1, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.15, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.2, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.3, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.4, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[1], 0.5, 1000)

    print "\n Third memory"
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.05, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.1, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.15, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.2, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.3, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.4, 1000)
    orthNet.getEspuriousRate(orthMemories, orthMemories[2], 0.5, 1000)
    

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
    #numbers = xrange(0,2**20)

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
    espuriousEncontrados = []
    for X in numbers:
        output = net.activate(np.copy(X))
        if not any((output == x).all() for x in memories):
            if not any((output == x).all() for x in espuriousEncontrados):
                espuriousEncontrados.append(output)
            count = count + 1

    count2 = 0
    espuriousEncontrados2 = []
    for X in numbers:
        output = orthNet.activate(np.copy(X))
        if not any((output == x).all() for x in orthMemories):
            if not any((output == x).all() for x in espuriousEncontrados2):
                espuriousEncontrados2.append(output)
            count2 = count2 + 1

    print "FOUND %d espirious states (%d occurrences) on %d patterns with orthogonal memories" % (len(espuriousEncontrados), count, len(numbers))
    print "FOUND %d espirious states (%d occurrences) on %d patterns non orthogonal memories" % (len(espuriousEncontrados2), count2, len(numbers))