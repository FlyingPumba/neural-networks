import numpy as np
import scipy as sci

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes, learning_rate, trainingset):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.lRate = learning_rate
        self.dataset = trainingset

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

        accumulated_error = 0
        epsilon = 0.5

        while True:
            # begin epoch
            print "The %d epoch has begun" % cant_epochs
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
                print auxMatrix

                Y = np.zeros(self.nOutput)
                for j in range(0, self.nOutput):
                    Y[j] = self.activation(auxMatrix[j])

                print Y

                # calculate the error
                E = Z - Y
                print "Error is: %s" % E
                accumulated_error = accumulated_error + (E[0]*E[1])

                # calculate the delta
                transposedX = X.reshape(X.shape+(1,))
                delta = self.lRate * np.multiply(transposedX, E)

                print "Delta error is: %s" % delta
                if(batch):
                    self.D = self.D + delta
                else:
                    self.W = self.W + delta
            
            if(batch):
                self.W = self.W + self.D

            cant_epochs = cant_epochs + 1
            if(cant_epochs >= max_epochs):
                print "REACHED MAX EPOCHS"
                break
            if(epsilon >= accumulated_error/cant_patterns):
                print "REACHED BETTER ERROR THAN EPSILON"
                break

            print "Final weight matrix is: %s" % self.W

    def activation(self, Y):
        # avoid returning -1
        aux = np.sign(Y)
        if(aux == -1):
            return 0
        else:
            return aux

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
    trainingset = np.array([
        [[0,0],[0,0]],
        [[0,1],[1,0]],
        [[1,0],[1,0]],
        [[1,1],[1,1]]
        ])
    a = PerceptronSimple(2, 2, 0.2, trainingset)
    testset = np.array([[0,0], [0,1], [1,0], [1,1]])
    for i in range(0,4):
        a.evaluate(testset[i])


if __name__ == "__main__":
    main()