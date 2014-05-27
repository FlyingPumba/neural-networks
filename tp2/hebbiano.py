import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd

class NoSupervisedNetwork():
    """Aprendizaje Hebbiano"""
    def __init__(self, learning_rate, dimens = []):
        self.nInput = 6
        self.nOutput = 4
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
        if sanger:
            max_epochs = 300
        else:
            max_epochs = 600

        while cant_epochs <= max_epochs:
            # begin a new epoch

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            for X in trainingset:
                Y = np.dot(X,W)

                dW = np.zeros(W.shape)
                for j in xrange(self.nOutput):
                    for i in xrange(self.nInput):
                        Xaux = 0
                        if sanger:
                            Q = j+1
                        else:
                            Q = self.nOutput

                        for k in xrange(Q):
                            Xaux = Xaux + Y[k]*W[i,k]
                        dW[i,j] = self.lRate * Y[j] * (X[i] - Xaux)
                W = W + dW

            cant_epochs = cant_epochs + 1

        self.W = W

    def plotWeights(self):
        plt.xlabel("Final weights")
        plt.imshow(self.W,interpolation='none', cmap=cm.gray)
        plt.colorbar()
        plt.show()

    def getInputArray(self):
        # create input array
        X = np.zeros(self.cantDimens)
        for j in xrange(self.cantDimens):
            X[j] = np.random.randint(-self.dimens[j], high=self.dimens[j])

        return X

# ========== MAIN ==========
# numpy print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

if __name__ == "__main__":
    dimens = [4,8,12,16,20,24]
    netOja = NoSupervisedNetwork(0.00001, dimens)
    netSanger = NoSupervisedNetwork(0.00001, dimens)

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

    netOja.plotWeights()
    netSanger.plotWeights()
