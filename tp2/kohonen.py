import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd

class NoSupervisedNetwork():
    """Mapas auto-organizados de Kohonen - Parte 1"""
    def __init__(self):
        self.nInput = 1
        self.nOutput = 20
        self.etaAlpha = 0.2
        self.sigmaAlpha  = 0.3

        self.etaHistory = []
        self.sigmaHistory = []

    def trainNetwork(self, dataset, stochastic=True):
        W = np.random.uniform(-0.1,0.1,size=(self.nInput, self.nOutput))

        cant_epochs = 1
        max_epochs = 1000

        while cant_epochs <= max_epochs:
            # begin a new epoch

            self.etaHistory.append(self.eta(cant_epochs))
            self.sigmaHistory.append(self.sigma(cant_epochs))

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            for X in trainingset:
                Y = self.activation(X,W)
                P = self.winner(Y)
                D = self.proxy(P, self.sigma(cant_epochs))
                dW = self.eta(cant_epochs) * (X.T - W) * D
                W = W + dW

            cant_epochs = cant_epochs + 1

        self.W = W

    def activation(self, X, W):
        Y = (W - X)**2
        Y = np.array(Y)
        return [True if x == min(Y[0]) else False for x in Y[0]]

    def eta(self, t):
        return t**(-self.etaAlpha)

    def sigma(self, t):
        return t**(-self.sigmaAlpha)

    def winner(self, y):
        p = -1
        for i in xrange(len(y)):
            if y[i]:
                p = i
                break
        return p

    def proxy(self, p, sigma):
        d = []
        for i in xrange(self.nOutput):
            aux = np.exp( (-(i-p)**2)/ 2*sigma**2)
            d.append(aux)
        return d

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

    def validateNetwork(self, validationset):
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
        #ax = sp1.axes()
        #ax.set_xticks(pos + (width / 2))
        #ax.set_xticklabels(pos)

        # show histogram
        sp1.bar(pos, frecuencias, width, color='r')
        sp1.set_title("Frecuencias de Activacion")
        #sp1.xlabel("Neurona")
        #sp1.ylabel("Frecuencia")
        sp1.set_xlabel('Neurona')
        sp1.set_ylabel('Frecuencia')

        # show eta and sigma history
        sp2 = plt.subplot(212)
        labelEta = 'eta (alpha: %.2f)' % self.etaAlpha
        sp2.plot(self.etaHistory, label=labelEta)
        labelSigma = 'sigma (alpha: %.2f)' % self.sigmaAlpha
        sp2.plot(self.sigmaHistory, label=labelSigma)
        sp2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True)

        fileName = 'kohonen1-%d.png' % np.random.randint(1000)
        plt.savefig(fileName, bbox_inches='tight')
        plt.show()

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    net = NoSupervisedNetwork()

    # generate the data and validation sets
    cant_patterns_training = 800
    cant_patterns_validation = 800
    cota = 50
    # var and mean for the normal distribution
    var = (2*cota)/8
    mean = 0

    # uniform training set
    uniform_set = []
    for i in xrange(cant_patterns_training):
        uniform_set.append(np.random.uniform(-cota,cota))

    # normal training set
    normal_set = []
    for i in xrange(cant_patterns_training):
        normal_set.append(np.random.normal(mean,var))

    # uniform validation set
    uniform_valset = []
    for i in xrange(cant_patterns_validation):
        uniform_valset.append(np.random.uniform(-cota,cota))

    # normal validation set
    normal_valset = []
    for i in xrange(cant_patterns_validation):
        normal_valset.append(np.random.normal(mean,var))

    net.trainNetwork(uniform_set)
    print "Final weights: %s" % net.W

    net.validateNetwork(normal_valset)
    #net.plotWeights()
