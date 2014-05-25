import numpy as np
import matplotlib.pyplot as plt
import data as data

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes, learning_rate, epsilon):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.lRate = learning_rate
        self.epsilon = epsilon

    def train_network(self, dataset, batch=False, stochastic=True):
        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]
        trainingset = np.copy(dataset)

        # create the Weight matrix (nInput+1 for the threshold)
        W = np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nOutput))

        cant_epochs = 0
        max_epochs = 1000

        self.errors_in_each_epoch = []
        self.appendEpochError = self.errors_in_each_epoch.append

        self.errors_in_each_epoch_sum = []
        self.appendEpochErrorSum = self.errors_in_each_epoch_sum.append

        while True:
            # begin a new epoch
            # to store the square error
            errors_in_each_pattern = []
            appendPatternError = errors_in_each_pattern.append

            errors_in_each_pattern_abs = []
            appendPatternErrorAbs = errors_in_each_pattern_abs.append

            # stochastic learning
            if(stochastic):
                trainingset = np.copy(dataset)
                np.random.shuffle(trainingset)

            if(batch):
                D = np.zeros((self.nInput+1, self.nOutput))

            for i in xrange(cant_patterns):
                X = self.getInputWithThreshold(trainingset[i,0])
                Y = np.tanh(np.dot(X,W))
                Z = trainingset[i,1]

                # calculate the error
                E = Z - Y
                
                appendPatternError(np.dot(E, E))
                appendPatternErrorAbs(np.sum(np.absolute(E)/2))

                # calculate the delta
                transposedX = np.array([X]).T
                delta = self.lRate * transposedX * E
                
                # learn !
                if(batch):
                    D = D + delta
                else:
                    W = W + delta
            
            if(batch):
                W = W + D

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