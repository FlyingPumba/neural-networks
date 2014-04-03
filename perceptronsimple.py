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

    def train_network(self, dataset, batch=False):
        print "Data set size is:"
        print "%s \n" % (dataset.shape,)

        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]

        # create D matrix for batch learning
        self.D = np.zeros(dataset.shape)

        # create the Weight matrix (nInput+1 for the threshold)
        self.W = np.random.uniform(-0.1,0.1,size=(self.nInput+1, self.nOutput))
        print "Initial weight matrix \n %s \n" % self.W

        cant_epochs = 0
        max_epochs = 1000

        self.errors_in_each_training = []

        while True:
            # begin epoch
            print "The %d epoch has begun \n" % cant_epochs
            accumulated_error = 0

            for i in range(0, cant_patterns):
                print "Training pattern %d" % i

                Y = self.evaluate(dataset[i,0])

                print "Expected output is: %s" % dataset[i,1]
                Z = dataset[i,1]

                # calculate the error
                E = Z - Y
                print "Error is: %s" % E
                 # XXX: in the following line,

                pattern_error = np.dot(E, E)
                self.errors_in_each_training.append(pattern_error)

                accumulated_error = accumulated_error + pattern_error

                # calculate the delta
                X = self.getInputWithThreshold(dataset[i,0])
                transposedX = X.reshape(X.shape+(1,))
                delta = self.lRate * np.multiply(transposedX, E) * (1 - (np.tanh(np.dot(X,self.W)))**2)
                print "Delta error is: \n%s\n" % delta

                # learn !
                if(batch):
                    self.D = self.D + delta
                else:
                    self.W = self.W + delta
            
            if(batch):
                self.W = self.W + self.D

            cant_epochs = cant_epochs + 1
            keep_going = True
            if(cant_epochs >= max_epochs):
                print "REACHED MAX EPOCHS\n"
                keep_going = False
            if(self.epsilon >= accumulated_error/cant_patterns):
                print "REACHED BETTER ERROR THAN EPSILON\n"
                keep_going = False

            if(keep_going == False):
                print "total epochs = %d\n" % cant_epochs
                print "last e = %.10f\n" % (accumulated_error/cant_patterns)
                break

        print "Final weight matrix is: \n%s\n" % self.W
        self.plotErrorThroughLearning(self.errors_in_each_training)

    def activation(self, Y):
        # use hiperbolic tangent
        return np.tanh(Y)

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
        print "Input: %s" % input
        X = self.getInputWithThreshold(input)

        # calculate the network output
        auxMatrix = np.dot(X,self.W)

        Y = self.activation(auxMatrix)

        print "Output: %s" % Y
        return Y

    def getInputWithThreshold(self, input):
        # create input array
        X = np.zeros(self.nInput+1)
        for j in range(0, self.nInput):
            X[j] = input[j]
        X[self.nInput] = -1

        return X

    def testNetwork(self, testset, testepsilon):
        print "\nTesting the network"
        cant_patterns = testset.shape[0]
        patterns_with_error = 0

        for i in range(0, cant_patterns):
            print "Testing pattern %d" % i

            Y = self.evaluate(testset[i,0])

            print "Expected output is: %s" % testset[i,1]
            Z = testset[i,1]

            # calculate the error
            E = Z - Y
            print "Error is: %s\n" % E

            absolute_errors = np.absolute(E)
            if(np.size(absolute_errors[absolute_errors>testepsilon]) != 0):
                patterns_with_error = patterns_with_error + 1

        print "There were %d errors over %d patterns" % (patterns_with_error, cant_patterns)
