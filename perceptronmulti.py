import numpy as np
import matplotlib.pyplot as plt
import data as data

class PerceptronMulti():
    """Perceptron Multicapa"""
    def __init__(self, cant_input_nodes, array_cant_hidden_nodes, cant_output_nodes, learning_rate, epsilon, trainingset):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.nHiddenNodes = array_cant_hidden_nodes
        self.nHiddenLayers = len(array_cant_hidden_nodes)
        self.lRate = learning_rate
        self.dataset = trainingset
        self.epsilon = epsilon

        self.train_network(self.dataset)

    def train_network(self, dataset, batch=False, stochastic=True):
        print "Data set size is:"
        print "%s \n" % (dataset.shape,)

        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]

        # create D matrix for batch learning
        self.D = np.zeros(dataset.shape)

        # create the Weight matrix (nInput+1 for the threshold)
        self.W = []
        self.W.append(np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nHiddenNodes[0])))
        for i in xrange(1, self.nHiddenLayers, 1):
            self.W.append(np.random.uniform(-0.1,0.1,size=(self.nHiddenNodes[i-1], self.nHiddenNodes[i])))
        self.W.append(np.random.uniform(-0.1,0.1,size=(self.nHiddenNodes[self.nHiddenLayers-1], self.nOutput)))

        cant_epochs = 0
        max_epochs = 1000

        self.errors_in_each_epoch = []
        self.errors_in_each_pattern = []

        while True:
            # begin epoch
            print "The %d epoch has begun \n" % cant_epochs
            self.errors_in_each_pattern = []

            # stochastic learning
            if(stochastic):
                np.random.shuffle(self.dataset)

            for i in xrange(cant_patterns):
                print "Training pattern %d" % i
                self.X = self.getInputWithThreshold(dataset[i,0])
                self.Y = self.evaluate(self.X)
                print "Output is: %s" % self.Y[-1]

                print "Expected output is: %s" % dataset[i,1]
                Z = dataset[i,1]

                self.G = self.backPropagation(self.Y,Z)

                # learn !
                if(batch):
                    self.D = self.updateWeights(self.W,self.G,self.lRate)
                else:
                    self.W = self.updateWeights(self.W,self.G,self.lRate)
            
            if(batch):
                self.W = self.W + self.D

            cant_epochs = cant_epochs + 1
            self.errors_in_each_epoch.append(np.max(self.errors_in_each_pattern))
            
            keep_going = True
            if(cant_epochs >= max_epochs):
                print "\nREACHED MAX EPOCHS\n"
                keep_going = False
            if(self.epsilon >= np.max(self.errors_in_each_pattern)):
                print "\nREACHED BETTER ERROR THAN EPSILON IN LAST EPOCH\n"
                keep_going = False

            if(keep_going == False):
                print "total epochs = %d\n" % cant_epochs
                print "last e = %.10f\n" % np.max(self.errors_in_each_pattern)
                break

        print "Final weight matrix is: \n%s\n" % self.W

    def plotErrorThroughLearning(self, errors_list):
        answer = ""
        while (answer != "y") & (answer != "n"):
            answer = raw_input("Do you wanna see the error through learning ? (y/n) ")

        if(answer == "y"):
            plt.plot(errors_list)
            plt.ylabel("network error")
            plt.xlabel("pattern number")
            plt.show()

    def evaluate(self, input):
        Y = []
        Y.append(self.activation(np.dot(self.X,self.W[0])))
        for i in xrange(1, self.nHiddenLayers, 1):
            Y.append(self.activation(np.dot(Y[i-1],self.W[i])))
        Y.append(self.activation(np.dot(Y[self.nHiddenLayers-1],self.W[self.nHiddenLayers])))
        #print "%s" % Y
        return Y

    def activation(self, Y):
        # use hiperbolic tangent
        return np.tanh(Y)

    def backPropagation(self, Y, Z):
        # calculate the error
        E = Z-Y[-1]

        pattern_error = np.dot(E, E)
        self.errors_in_each_pattern.append(pattern_error)

        L = self.nHiddenLayers
        G = []

        for i in xrange(L, 0, -1):
            transposedInput = Y[i-1].reshape(Y[i-1].shape+(1,))
            d = (1 - (np.tanh(np.dot(Y[i-1],self.W[i])))**2)
            aux = d * transposedInput * E 
            G.append(aux)
            E = np.dot(aux,self.W[i].T)
        return G

    def updateWeights(self, W, G, eta):
        for i in xrange(1, self.nHiddenLayers, 1):
            W[i] = W[i] + eta*G[i]
        return W

    def getInputWithThreshold(self, input):
        # create input array
        X = np.zeros(self.nInput+1)
        for j in xrange(self.nInput):
            X[j] = input[j]
        X[self.nInput] = -1

        return X

    def testNetwork(self, testset, testepsilon):
        print "\nTesting the network"
        cant_patterns = testset.shape[0]
        patterns_with_error = 0

        for i in xrange(cant_patterns):
            print "\nTesting pattern %d" % (i+1)

            Y = self.evaluate(testset[i,0])
            print "Output is: %s" % self.Y[-1]
            
            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = Z - Y[-1]
            print "Error is: %s" % E

            absolute_errors = np.absolute(E)
            if(np.size(absolute_errors[absolute_errors>testepsilon]) != 0):
                print "-----> WRONG OUTPUT"
                patterns_with_error = patterns_with_error + 1

        print "\nThere were %d errors over %d patterns" % (patterns_with_error, cant_patterns)
