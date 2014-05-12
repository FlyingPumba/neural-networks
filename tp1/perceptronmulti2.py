import numpy as np
import matplotlib.pyplot as plt
import data as data

class PerceptronMulti():
    """Perceptron Multicapa"""
    def __init__(self, cant_input_nodes, cant_hidden_nodes, cant_output_nodes, learning_rate, epsilon):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.nHiddenNodes = cant_hidden_nodes
        self.lRate = learning_rate
        self.epsilon = epsilon
        #momentum parameter
        self.alpha = 0.5
        #dlr parameters
        self.a = 0.1
        self.b = 0.1
        self.epochsToCorrect = 10
        self.currentStrikeOfBadEpochs = 0

    def train_network(self, dataset, stochastic=True, momentum=False, dlr=False):
        print "Data set size is:"
        print "%s \n" % (dataset.shape,)

        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]
        trainingset = np.copy(dataset)

        # create the Weight matrix (nInput+1 for the threshold)
        W = []
        W.append(np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nHiddenNodes)))
        W.append(np.random.uniform(-0.1,0.1,size=(self.nHiddenNodes+1, self.nOutput)))

        print "W: %s" % W 

        if(dlr):
            Wdlr = []

        cant_epochs = 0
        max_epochs = 5000

        self.errors_in_each_epoch = []
        self.appendEpochError = self.errors_in_each_epoch.append

        self.errors_in_each_epoch_sum = []
        self.appendEpochErrorSum = self.errors_in_each_epoch_sum.append

        while True:
            # begin epoch
            print "\nThe %d epoch has begun \n" % cant_epochs
            errors_in_each_pattern = []
            appendPatternError = errors_in_each_pattern.append
            errors_in_each_pattern_abs = []
            appendPatternErrorAbs = errors_in_each_pattern_abs.append

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            if(momentum):
                Gm = []

            for i in xrange(cant_patterns):
                X = trainingset[i,0]
                print "Training pattern %d: %s" % (i,X)
                Y = self.evaluate(X, W)
                print "Output is: %s" % Y[1]

                print "Expected output is: %s" % dataset[i,1]
                Z = trainingset[i,1]

                E = Z-Y[1]
                pattern_error = np.dot(E, E)
                appendPatternError(pattern_error)
                appendPatternErrorAbs(np.sum(np.absolute(E)/2))

                G = self.backPropagation(X,Y,Z,W, self.lRate)

                #learn !
                W = self.updateWeights(W,G)
                #print "W: %s" % W

                if(momentum):
                    if(len(Gm) > 0):
                        W = self.addMomentum(W, Gm, self.alpha)
                    Gm = G

                if(dlr and len(self.errors_in_each_epoch) > self.epochsToCorrect):
                    gap = self.errors_in_each_epoch[-1] - self.errors_in_each_epoch[-(self.epochsToCorrect+1)]
                    #print "gap: %.10f" % gap
                    if(gap < 0) :
                        #speed up !
                        self.lRate = self.lRate + self.a
                        self.currentStrikeOfBadEpochs = 0
                        #print "new lRate: %.5f" % self.lRate
                    elif(gap > 0):
                        if(self.currentStrikeOfBadEpochs == 0):
                            Wdlr = W
                            self.currentStrikeOfBadEpochs = self.currentStrikeOfBadEpochs + 1
                        elif(self.currentStrikeOfBadEpochs == self.epochsToCorrect):
                            #slow down and rollback
                            self.lRate = self.lRate - (self.lRate * self.b)
                            W = Wdlr
                            self.currentStrikeOfBadEpochs = 0
                        else:
                            self.currentStrikeOfBadEpochs = self.currentStrikeOfBadEpochs + 1

            cant_epochs = cant_epochs + 1
            self.appendEpochError(np.max(errors_in_each_pattern))
            self.appendEpochErrorSum(np.sum(errors_in_each_pattern_abs))
            
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
            #plt.title('Title of the graphic')
            plt.show()

    def evaluate(self, input, weights):
        #print "W to evaluate: %s" % weights
        Y = []
        Y.append(np.tanh(np.dot(self.addBias(input),weights[0])))
        Y.append(np.tanh(np.dot(self.addBias(Y[0]),weights[1])))
        #print "Y: %s" % Y
        return Y

    def backPropagation(self,X, Y, Z, weights, eta):
        G = []

        E1 = Z-Y[1]
        D1 = E1 * (1 - Y[1]**2)
        delta1 = eta * np.outer(self.addBias(Y[0]), D1)

        E0 = self.removeBias(np.dot(D1,weights[1].T))
        D0 = E0 * (1 - Y[0]**2)
        delta0 = eta * np.outer(self.addBias(X), D0)

        G.append(delta0)
        G.append(delta1)

        return G

    def updateWeights(self, W, G):
        W[0] = W[0] + G[0]
        W[1] = W[1] + G[1]
        return W

    def addMomentum(self, W, Gm, alpha):
        W[0] = W[0] + alpha*Gm[0]
        W[1] = W[1] + alpha*Gm[1]
        return W

    def addBias(self, input):
        X = np.resize(input, np.size(input)+1)
        X[-1] = 1

        return X

    def removeBias(self, input):
        return input[:-1]

    def testNetwork(self, testset, testepsilon):
        print "\nTesting the network"
        cant_patterns = testset.shape[0]
        patterns_with_error = 0

        for i in xrange(cant_patterns):
            X = testset[i,0]
            print "\nTesting pattern %d: %s" % ((i+1), X)
            Y = self.evaluate(X, self.W)
            print "Output is: %s" % Y[1]
            
            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = Z - Y[1]
            print "Error is: %s" % E

            absolute_error = np.absolute(E)/2
            sum_error = np.sum(absolute_error)
            if(sum_error>testepsilon):
                print "-----> WRONG OUTPUT. The sum error for this pattern is: %.5f" % sum_error
                patterns_with_error = patterns_with_error + 1

        print "\nThere were %d errors over %d patterns" % (patterns_with_error, cant_patterns)
        return patterns_with_error
