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
        print self.epsilon

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

        errors_in_each_training = []

        while True:
            # begin epoch
            print "The %d epoch has begun \n" % cant_epochs
            accumulated_error = 0

            for i in range(0, cant_patterns):
                print "Training pattern %d. Input: %s -> Expected output: %s" % (i, dataset[i,0], dataset[i,1])
                Z = dataset[i,1]

                # create input array
                X = np.zeros(self.nInput+1)
                for j in range(0, self.nInput):
                    X[j] = dataset[i,0,j]
                X[self.nInput] = -1

                # calculate the network output
                auxMatrix = np.dot(X,self.W)

                Y = np.zeros(self.nOutput)
                for j in range(0, self.nOutput):
                    Y[j] = self.activation(auxMatrix[j])

                print "Network output is: %s" % Y

                # calculate the error
                E = Z - Y
                print "Error is: %s" % E
                max_error = np.amax(E)
                errors_in_each_training.append(max_error)

                accumulated_error = accumulated_error + max_error * max_error

                # calculate the delta
                transposedX = X.reshape(X.shape+(1,))
                delta = self.lRate * np.multiply(transposedX, E)
                print "Delta error is: \n%s\n" % delta

                # learn !
                if(batch):
                    self.D = self.D + delta
                else:
                    self.W = self.W + delta
            
            if(batch):
                self.W = self.W + self.D

            cant_epochs = cant_epochs + 1
            if(cant_epochs >= max_epochs):
                print "REACHED MAX EPOCHS\n"
                break
            if(self.epsilon >= accumulated_error/cant_patterns):
                print "REACHED BETTER ERROR THAN EPSILON\n"
                break

        print "Final weight matrix is: \n%s\n" % self.W
        self.plotErrorThroughLearning(errors_in_each_training)

    def activation(self, Y):
        # avoid returning -1
        aux = np.sign(Y)
        if(aux == -1):
            return 0
        else:
            return aux

    def plotErrorThroughLearning(self, errors_list):
        answer = ""
        while (answer != "y") & (answer != "n"):
            answer = raw_input("Do you wanna see the error throuh learning ? (y/n)")

        if(answer == "y"):
            plt.plot(errors_list)
            plt.ylabel("network error")
            plt.xlabel("pattern number")
            plt.show()

    def evaluate(self, input):
        # create input array
        X = np.zeros(self.nInput+1)
        for j in range(0, self.nInput):
            X[j] = input[j]
        X[self.nInput] = -1

        # calculate the network output
        auxMatrix = np.dot(X,self.W)

        Y = np.zeros(self.nOutput)
        for j in range(0, self.nOutput):
            Y[j] = self.activation(auxMatrix[j])

        print "The network output for %s is: %s" % (input, Y)

def main():
    
    # input nodes: 2, output nodes: 2, lRate: 0.2, epsilon: 0.1
    a = PerceptronSimple(2, 2, 0.2, 0.1, data.ANDOR.trainingset)

    
    print "\nTesting the network"
    for i in range(0,4):
        a.evaluate(data.ANDOR.testset[i])

    # print some nice graphics
    #plt.plot(testset)
    #plt.ylabel("test set")
    #plt.show()


if __name__ == "__main__":
    main()