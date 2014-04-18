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
        self.epsilon = epsilon
        #momentum parameter
        self.alpha = 0.01
        #dlr parameters
        self.a = 0.1
        self.b = 0.1
        self.gapOfErrorsToCorrect = 3

        self.train_network(trainingset, batch=False, stochastic=False, momentum=False, dlr=False)

    def train_network(self, dataset, batch=False, stochastic=True, momentum=False, dlr=False):
        print "Data set size is:"
        print "%s \n" % (dataset.shape,)

        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]

        # create the Weight matrix (nInput+1 for the threshold)
        W = []
        W.append(np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nHiddenNodes[0])))
        for i in xrange(1, self.nHiddenLayers, 1):
            W.append(np.random.uniform(-0.1,0.1,size=(self.nHiddenNodes[i-1], self.nHiddenNodes[i])))
        W.append(np.random.uniform(-0.1,0.1,size=(self.nHiddenNodes[self.nHiddenLayers-1], self.nOutput)))

        if(dlr):
            Wdlr = []

        cant_epochs = 0
        max_epochs = 1000

        self.errors_in_each_epoch = []
        self.appendEpochError = self.errors_in_each_epoch.append

        self.errors_in_each_epoch_sum = []
        self.appendEpochErrorSum = self.errors_in_each_epoch_sum.append

        while True:
            # begin epoch
            print "\nThe %d epoch has begun \n" % cant_epochs
            errors_in_each_pattern = []
            appendPatternError = errors_in_each_pattern.append

            # stochastic learning
            if(stochastic):
                np.random.shuffle(dataset)

            if(batch):
                D = []
                D.append(np.zeros((self.nInput+1, self.nHiddenNodes[0])))
                for i in xrange(1, self.nHiddenLayers, 1):
                    D.append(np.zeros((self.nHiddenNodes[i-1], self.nHiddenNodes[i])))
                D.append(np.zeros((self.nHiddenNodes[self.nHiddenLayers-1], self.nOutput)))

            if(momentum):
                Gm = []

            if(dlr and cant_epochs % self.gapOfErrorsToCorrect == 0):
                #update the learning rate
                if(len(self.errors_in_each_epoch) > self.gapOfErrorsToCorrect):
                    gap = self.errors_in_each_epoch[-1] - self.errors_in_each_epoch[-(self.gapOfErrorsToCorrect+1)]
                    print "gap: %.10f" % gap
                    if(gap < 0) :
                        #speed up !
                        self.lRate = self.lRate + self.a
                        print "new lRate: %.5f" % self.lRate
                    elif(gap > 0):
                        #slow down
                        self.lRate = self.lRate - (self.lRate * self.b)
                        print "new lRate: %.5f" % self.lRate
                        #erase last epochs
                        W = Wdlr

                    #raw_input()

            for i in xrange(cant_patterns):
                print "Training pattern %d" % i
                X = self.getInputWithThreshold(dataset[i,0])
                Y = self.evaluate(X, W)
                print "Output is: %s" % Y[-1]

                print "Expected output is: %s" % dataset[i,1]
                Z = dataset[i,1]

                E = Z-Y[-1]
                pattern_error = np.dot(E, E)
                appendPatternError(pattern_error)

                W = self.backPropagation(X,Y,Z,W)

                # learn !
                #if(batch):
                    #D = self.updateWeights(D,G,self.lRate)
                #else:
                    #W = self.updateWeights(W,G,self.lRate)

                if(momentum):
                    if(len(Gm) > 0):
                        W = self.addMomentum(W, Gm, self.alpha)
                    Gm = G
            
            if(batch):
                for i in xrange(0, self.nHiddenLayers+1, 1):
                    W[i] = W[i] + D[i]

            if(dlr and cant_epochs % self.gapOfErrorsToCorrect == 0):
                print "saving W on epoch: %d" % cant_epochs
                #raw_input()
                Wdlr = W

            cant_epochs = cant_epochs + 1
            self.appendEpochError(np.max(errors_in_each_pattern))
            self.appendEpochErrorSum(np.sum(errors_in_each_pattern))
            
            keep_going = True
            if(cant_epochs >= max_epochs):
                print "\nREACHED MAX EPOCHS\n"
                keep_going = False
            if(self.epsilon >= np.max(errors_in_each_pattern)):
                print "\nREACHED BETTER ERROR THAN EPSILON IN LAST EPOCH\n"
                keep_going = False

            if(keep_going == False):
                self.W = W
                print "total epochs = %d\n" % cant_epochs
                print "last e = %.10f\n" % np.max(errors_in_each_pattern)
                break

        print "Final weight matrix is: \n%s\n" % self.W

    def plotErrorThroughLearning(self):
        answer = ""
        while (answer != "y") & (answer != "n"):
            answer = raw_input("Do you wanna see the error through learning ? (y/n) ")

        if(answer == "y"):
            plt.plot(self.errors_in_each_epoch)
            plt.plot(self.errors_in_each_epoch_sum)
            plt.ylabel("network error")
            plt.xlabel("epoch number")
            plt.show()

    def evaluate(self, input, weights):
        Y = []
        append = Y.append
        append(np.tanh(np.dot(input,weights[0])))
        for i in xrange(1, self.nHiddenLayers, 1):
            append(np.tanh(np.dot(Y[i-1],weights[i])))
        append(np.tanh(np.dot(Y[self.nHiddenLayers-1],weights[self.nHiddenLayers])))

        return Y

    def backPropagation(self,X, Y, Z, weights):
        # calculate the error
        E = Z-Y[-1]

        L = self.nHiddenLayers
        G = []
        append = G.append

        for i in xrange(L, 0, -1):
            derivative = 1 - np.tanh(Y[i])**2
            transposedInput = np.array([Y[i-1]]).T
            aux = E * derivative
            delta = transposedInput * aux
            weights[i] = weights[i] + delta
            #append(delta)
            E = np.dot(aux,weights[i].T)

        derivative = 1 - np.tanh([Y[0]])**2
        transposedInput = np.array([X]).T
        aux = E * derivative
        delta = transposedInput * aux
        weights[0] = weights[0] + delta
        #append(delta)

        return weights

    def updateWeights(self, W, G, eta):
        #remember that G is backwards
        for i in xrange(0, self.nHiddenLayers+1, 1):
            W[i] = W[i] + eta*G[self.nHiddenLayers-i]
        return W

    def addMomentum(self, W, Gm, alpha):
        #remember that Gm is backwards
        for i in xrange(0, self.nHiddenLayers+1, 1):
            W[i] = W[i] + alpha*Gm[self.nHiddenLayers-i]
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
            X = self.getInputWithThreshold(testset[i,0])
            Y = self.evaluate(X, self.W)
            print "Output is: %s" % Y[-1]
            
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
