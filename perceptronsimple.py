import numpy as np
import matplotlib.pyplot as plt
import data as data

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes, learning_rate, epsilon, trainingset):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.lRate = learning_rate
        self.dataset = trainingset
        self.epsilon = epsilon

        self.train_network(self.dataset)

    def train_network(self, dataset, batch=False, stochastic=True):
        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]

        # create D matrix for batch learning
        self.D = np.zeros(dataset.shape)

        # create the Weight matrix (nInput+1 for the threshold)
        self.W = np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nOutput))

        cant_epochs = 0
        max_epochs = 10000

        self.errors_in_each_epoch = []
        self.appendEpochError = self.errors_in_each_epoch.append        

        while True:
            # begin epoch
            print "The %d epoch has begun \n" % cant_epochs
            errors_in_each_pattern = []
            appendPatternError = errors_in_each_pattern.append

            # stochastic learning
            if(stochastic):
                np.random.shuffle(self.dataset)

            for i in xrange(cant_patterns):
                self.X = self.getInputWithThreshold(dataset[i,0])
                self.evaluate()
                self.Z = dataset[i,1]

                # calculate the error
                E = self.Z - self.Y
                
                appendPatternError(np.dot(E, E))

                # calculate the delta
                transposedX = self.X.reshape(self.X.shape+(1,))
                delta = self.lRate * np.multiply(transposedX, E) * (1 - (np.tanh(np.dot(self.X,self.W)))**2)

                # learn !
                if(batch):
                    self.D = self.D + delta
                else:
                    self.W = self.W + delta
            
            if(batch):
                self.W = self.W + self.D

            cant_epochs = cant_epochs + 1
            self.appendEpochError(np.max(errors_in_each_pattern))
            
            keep_going = True
            if(cant_epochs >= max_epochs):
                print "\nREACHED MAX EPOCHS\n"
                keep_going = False
            if(self.epsilon >= np.max(errors_in_each_pattern)):
                print "\nREACHED BETTER ERROR THAN EPSILON IN LAST EPOCH\n"
                keep_going = False

            if(keep_going == False):
                print "total epochs = %d\n" % cant_epochs
                print "last e = %.10f\n" % np.max(errors_in_each_pattern)
                break

    def plotErrorThroughLearning(self, errors_list):
        answer = ""
        while (answer != "y") & (answer != "n"):
            answer = raw_input("Do you wanna see the error through learning ? (y/n) ")

        if(answer == "y"):
            plt.plot(errors_list)
            plt.ylabel("network error")
            plt.xlabel("pattern number")
            plt.show()

    def evaluate(self):
        # calculate the network output
        self.Y = np.tanh(np.dot(self.X,self.W))

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
            self.X = self.getInputWithThreshold(testset[i,0])
            self.evaluate()

            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = self.Z - self.Y
            print "Error is: %s" % E

            absolute_errors = np.absolute(E)
            if(np.size(absolute_errors[absolute_errors>testepsilon]) != 0):
                print "-----> WRONG OUTPUT"
                patterns_with_error = patterns_with_error + 1

        print "\nThere were %d errors over %d patterns" % (patterns_with_error, cant_patterns)
