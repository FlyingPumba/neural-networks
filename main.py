import numpy as np
import scipy as sci

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes, learning_rate):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes
        self.lRate = learning_rate

        dataset = np.array([
            [[0,0],[0,0]],
            [[0,1],[1,0]],
            [[1,0],[1,0]],
            [[1,1],[1,1]]
            ])

        self.train_network(dataset)

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

        # begin epoch
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

            # calculate the delta
            transposedX = X.reshape(X.shape+(1,))
            delta = self.lRate * np.multiply(transposedX, E)

            print delta
            if(batch):
                self.D = self.D + delta
            else:
                self.W = self.W + delta
        
        if(batch):
            self.W = self.W + self.D

        print self.W

    def activation(self, Y):
        # avoid returning -1
        aux = np.sign(Y)
        if(aux == -1):
            return 0
        else:
            return aux

def main():
    a = PerceptronSimple(2, 2, 0.2)

if __name__ == "__main__":
    main()