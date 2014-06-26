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

    def activate(self, X, memories, temp, dist, max, hamming = False, plotEnergy=False):
        S = X
        Saux = np.zeros(self.cantNeuronas)
        Eh = []
        plt.ion()
        vueltas = 0

        currentDist = min(self.dist(Saux, i, hamming) > dist for i in memories)
        #print currentDist

        while all(self.dist(Saux, i, hamming) > dist for i in memories) and max >= 0:
            
            I = np.random.permutation(self.cantNeuronas)
            for i in I:
                if temp != 0 :
                    S[i] = np.sign(self.sigmoid(np.dot(S, self.W[:,i]), temp) - np.random.uniform(0,1))
                else:
                    S[i] = np.sign(np.dot(S, self.W[:,i]))

            currentDist = min(self.dist(Saux, i) > dist for i in memories)
            #print currentDist

            E = self.energy(S,self.W)
            #print "E: %s" % E
            Eh.append(E)

            if plotEnergy:
                self.plotEnergy(Eh)
            
            Saux = np.copy(S)
            vueltas += 1
            max -= 1
        
        #print vueltas
        if plotEnergy:
            raw_input()

        plt.ioff()
        return Saux

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
    
    def dist(a, b, hamming=False):
        if hamming:
            return distHamming(a, b)
        else:
            return distEuclidean(a, b)

    def distEuclidean(a, b):
        return np.sqrt(np.sum((b - a)**2))
   
    def distHamming(a, b):
        dist = 0
        for i in xrange(len(a)) :
            if a[i] != b[i]:
                dist += 1
        return dist
    
    def cMem(a, mem, hamming = False):
        mindist = dist(a, mem[0], hamming)
        minmem = mem[0]
        for i in mem:
            if dist(a, i, hamming) <= mindist:
                mindist = dist(a, i, hamming)
                minmem = i
                
        return minmem

    # temperature
    temp = 0
    maxTemp = 0.6
    inc = 0.1

    # distance
    distances = [5, 10, 15]
    totalDists = 3
    maxActIt = [9000, 500, 100] # Max itereations for the activation function

    # noise
    noise = [0.05, 0.10, 0.15, 0.20]
    
    # generate the memories
    cantMemorias = 10
    memories = []
 
    while cantMemorias > 0 :
        memories.append(np.round(np.random.sample(100)) * 2 - 1)
        cantMemorias -= 1
    
    cantMemorias = 10
    
    net.createWeights(memories)

    # ========== VALIDATION ==========
    # print [i for i in memories if self.dist(Saux, i, hamming) <= dist]
    # test that for all the memories, the ouput of the activation is the same
    
    print "\n VALIDATION with original memories\n"
    while temp <= maxTemp :
        curDistIndex = 0
        while curDistIndex < totalDists :
            proportions = []
            for X in memories:
                output = net.activate(np.copy(X), memories, temp, distances[curDistIndex], maxActIt[curDistIndex], hamming = True, plotEnergy=False)
             
                proportions.append((cMem(output, memories, hamming = True) == X).all())
            
            print "Current distance: " + str(distances[curDistIndex]) + " Current temperature: " + str(temp) + " Correctly detected memories (in percent): " + str((proportions.count(True) * 100.0) / cantMemorias)
            curDistIndex += 1
        temp += inc
    
    print "\n NOISE tests"
    for n in noise:
        temp = 0
        print "Current noise level: ", n
        while temp <= maxTemp :
            curDistIndex = 0
            while curDistIndex < totalDists :
                proportions = []
                for X in memories:
                    output = net.activate(np.copy(du.getPatternWithNoise(X, n)), memories, temp, distances[curDistIndex], maxActIt[curDistIndex], hamming = True, plotEnergy=False)
                    proportions.append((cMem(output, memories, hamming = True) == X).all())
                    break
                print "Current distance: " + str(distances[curDistIndex]) + " Current temperature: " + str(temp) + " Correctly detected memories (in percent): " + str((proportions.count(True) * 100.0) / cantMemorias)
                curDistIndex += 1
            temp += inc
    
    print "\n RANDOMLY generated patterns"
    temp = 0
    randPat = []
    cantRandPat = 10
    while cantRandPat > 0 :
        randPat.append(np.round(np.random.sample(100)) * 2 - 1)
        cantRandPat -= 1
    cantRandPat = 10
    #print randPat
    while temp <= maxTemp :
        curDistIndex = 0
        while curDistIndex < totalDists :
            proportions = []
            for X in randPat:
                output = net.activate(np.copy(X), memories, temp, distances[curDistIndex], maxActIt[curDistIndex], hamming = True, plotEnergy=False)
                proportions.append((cMem(output, randPat, hamming = True) == X).all())
            print "Current distance: " + str(distances[curDistIndex]) + " Current temperature: " + str(temp) + " Correctly detected memories (in percent): " + str((proportions.count(True) * 100.0) / cantRandPat)
            curDistIndex += 1
        temp += inc
    
    print "\n SPURIOUS states\n"
    temp = 0
    combinations = [3, 5, 7]
    spurious = []
    for c in combinations:
        tSpurious = []
        for i in range(c):
            tSpurious.append(rnd.choice(memories))
        spurious.append(sum(tSpurious))

    while temp <= maxTemp :
        curDistIndex = 0
        while curDistIndex < totalDists :
            proportions = []
            for X in spurious:
                output = net.activate(np.copy(X), memories, temp, distances[curDistIndex], maxActIt[curDistIndex], hamming = True, plotEnergy=False)
                proportions.append((cMem(output, spurious, hamming = True) == X).all())
            
            print "Current distance: " + str(distances[curDistIndex]) + " Current temperature: " + str(temp) + " Correctly detected memories (in percent): " + str((proportions.count(True) * 100.0) / cantMemorias)
            curDistIndex += 1
        temp += inc
    