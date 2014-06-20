import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    def validateNetwork(self, validationset, keepDrawing=True):
        frecuencias = [0]*self.nOutput
        for X in validationset:
            Y = self.activation(X,self.W)
            indice = Y.index(True)
            #print "X: %d -> Neurona: %d" % (X, indice)
            frecuencias[indice] += 1

        pos = np.arange(self.nOutput)
        width = 1.0     # gives histogram aspect to the bar diagram
        # show label for all bins
        sp1 = plt.subplot(211)

        # show histogram

        # don't let pyplot remove the values from the margins that are zero
        if frecuencias[0] == 0:
                frecuencias[0] = 0.00001
        if frecuencias[self.nOutput-1] == 0:
                frecuencias[self.nOutput-1] = 0.00001

        sp1.bar(pos, frecuencias, width, color='r')
        sp1.set_title("Frecuencias de Activacion")
        sp1.set_xlabel('Neurona')
        sp1.set_ylabel('Frecuencia')

        # show eta and sigma history
        sp2 = plt.subplot(212)
        labelEta = 'eta (alpha-ord: %.2f alpha-conv: %.2f)' % (self.etaAlphaOrdenamiento, self.etaAlphaConvergencia)
        sp2.plot(self.etaHistory, label=labelEta)
        labelSigma = 'sigma (alpha: %.2f)' % self.sigmaAlpha
        sp2.plot(self.sigmaHistory, label=labelSigma)
        sp2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True)

        if keepDrawing:
            plt.draw()
            plt.clf()
        else:
            fileName = 'kohonen1-%d.png' % np.random.randint(1000)
            print fileName
            figure = plt.gcf() # get current figure
            figure.set_size_inches(10, 8) #this will give us a 800x600 image
            # when saving, specify the DPI
            plt.savefig(fileName, bbox_inches='tight', dpi = 100)
            plt.close()

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
    for i in xrange(cantMemorias):
        mem = np.random.randint(2, size=net.cantNeuronas)
        mem = mem * 2 - 1
        memories.append(mem)

    print "Memories: %s" % memories
    net.createWeights(memories)
    print "Weights: %s" % net.W
    
    print "\nTesting memory 1 (with one bit changed). \nOriginal memory is: %s" % memories[0]
    validation = np.copy(memories[0])
    validation[3] *= -1;
    output = net.activate(validation)
    print "\nNetwork output: %s" % output
    print "Equals original memory: %s" % (output ==memories[0]).all()
    
    print "\nTesting memory 1 (with 5 bit changed). \nOriginal memory is: %s" % memories[0]
    validation = np.copy(memories[0])
    validation[3] *= -1;
    validation[6] *= -1;
    validation[9] *= -1;
    validation[12] *= -1;
    validation[15] *= -1;
    output = net.activate(validation)
    print "\nNetwork output: %s" % output
    print "Equals original memory: %s" % (output ==memories[0]).all()
    
    print "\nTesting memory 2 (with 5 bit changed). \nOriginal memory is: %s" % memories[1]
    validation = np.copy(memories[1])
    validation[3] *= -1;
    validation[6] *= -1;
    validation[9] *= -1;
    validation[12] *= -1;
    validation[15] *= -1;
    output = net.activate(validation)
    print "\nNetwork output: %s" % output
    print "Equals original memory: %s" % (output ==memories[1]).all()
