import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class NoSupervisedNetwork():
    """Aprendizaje Hebbiano"""
    def __init__(self, learning_rate, dimens = []):
        self.nInput = 2
        self.nOutput = 2
        self.lRate = learning_rate

        # problem parameters
        if dimens == []:
            self.cantDimens = self.nInput
            self.dimens = rnd.sample(range(1,100), self.cantDimens)
        else:
            self.cantDimens = len(dimens)
            self.dimens = dimens

    # the learning rule: if Sanger == false, we'll use OjaM
    def trainNetwork(self, dataset, stochastic=True, sanger = True):
        W = np.random.uniform(-0.1,0.1,size=(self.nInput, self.nOutput))

        cant_epochs = 0
        max_epochs = 10000

        while cant_epochs >= max_epochs:
            # begin a new epoch

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(self.dataset)
                np.random.shuffle(trainingset)

            for X in trainingset:
                Y = np.dot(X,W)

                dW = np.zeros(W.shape)
                for j in xrange(self.nOutput):
                    for i in xrange(self.nInput):
                        Xaux = 0
                        if sanger:
                            Q = j
                        else:
                            Q = self.nOutput

                        for k in xrange(Q):
                            Xaux = Xaux + Y[k]*W[i,k]
                        dW = self.lRate * (X[i] - Xaux)
                W = W + dW

            cant_epochs = cant_epochs + 1

        self.W = W

    def plotWeights(self):
        plt.plot(self.W, label="Weights")
        #plt.ylabel("network error")
        plt.xlabel("Final weights")
        plt.show()

    def plotDataset(self):
        # unpacking argument lists
        plt.plot(*zip(*self.dataset), marker='o', color='r', ls='')

        aux = np.copy(self.W)
        for i in xrange(self.cantDimens):
            aux[i] = aux[i] * self.dimens[i]

        print "Aux: %s" % aux
        plt.plot(*zip(*aux), label="Weights", marker='x', color='g', ls='')

        #plot (0,0)
        plt.plot([[0,0]], marker='+', color='b', ls='')
        
        plt.xlabel("Final weights")
        plt.show()

    def getInputArray(self):
        # create input array
        X = np.zeros(self.cantDimens)
        for j in xrange(self.cantDimens):
            X[j] = np.random.randint(-self.dimens[j], high=self.dimens[j])

        return X

    def testNetwork(self, testset, testepsilon):
        print "\nTesting the network"
        cant_patterns = testset.shape[0]
        patterns_with_error = 0

        for i in xrange(cant_patterns):
            print "\nTesting pattern %d" % (i+1)
            X = self.getInputWithThreshold(testset[i,0])
            Y = np.tanh(np.dot(X,self.W))
            print "Output is: %s" % Y

            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = Z - Y
            print "Error is: %s" % E

            absolute_error = np.absolute(E)/2
            sum_error = np.sum(absolute_error)
            if(sum_error>testepsilon):
                print "-----> WRONG OUTPUT. The sum error for this pattern is: %.5f" % sum_error
                patterns_with_error = patterns_with_error + 1

        print "\nThere were %d errors over %d patterns" % (patterns_with_error, cant_patterns)

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    netOja = NoSupervisedNetwork(0.2)
    netSanger = NoSupervisedNetwork(0.2, netOja.dimens)
    netSanger2 = NoSupervisedNetwork(0.2, netOja.dimens)

    # generate the dataset
    cant_patterns = 200
    dataset = []
    for i in xrange(cant_patterns):
        dataset.append(netOja.getInputArray())

    print "Dimens: %s" % netOja.dimens

    netOja.trainNetwork(dataset, sanger=False)
    print "Final weights oja: %s" % netOja.W

    netSanger.trainNetwork(dataset)
    print "Final weights sanger: %s" % netSanger.W

    netSanger2.trainNetwork(dataset)
    print "Final weights sanger2: %s" % netSanger2.W
    #net.plotWeights()

    #net.plotDataset()

    #print "Producto de los pesos: %s" % (net.W[0]*net.W[1])

    #var = [net.cantDimens]
    #mean = [net.cantDimens]
