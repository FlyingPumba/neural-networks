import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd

class NoSupervisedNetwork():
    """Mapas auto-organizados de Kohonen - Parte 1"""
    def __init__(self):
        self.nInput = 1
        self.nOutput = 15
        self.etaAlpha = 0.1
        self.sigmaAlpha  = 0.01

    def trainNetwork(self, dataset, stochastic=True):
        W = np.random.uniform(-0.1,0.1,size=(self.nInput, self.nOutput))

        cant_epochs = 1
        max_epochs = 100

        while cant_epochs <= max_epochs:
            # begin a new epoch

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
        Y = (W - X.T)**2
        Y = np.array(Y)
        return [True if x == min(Y[0]) else False for x in Y[0]]

    def eta(self, t):
        return t**(-self.etaAlpha )

    def sigma(self, t):
        return t**(-self.sigmaAlpha )

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
        plt.plot(self.W, label="Weights")
        #plt.ylabel("network error")
        plt.xlabel("Final weights")
        plt.show()

    def plotDatasetWithWeights(self, dataset):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.show()

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    net = NoSupervisedNetwork()

    # generate the dataset
    cant_patterns = 200
    cota = 50
    dataset = []
    for i in xrange(cant_patterns):
        dataset.append(np.random.uniform(-cota,cota))

    net.trainNetwork(dataset)
    print "Final weights: %s" % net.W

    #netOja.plotDatasetWithWeights(dataset)
