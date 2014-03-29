import numpy as np
import scipy as sci

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self, cant_input_nodes, cant_output_nodes):
        self.nInput = cant_input_nodes
        self.nOutput = cant_output_nodes

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

        # create the Weight matrix (nOutput+1 for the threshold)
        self.W = np.random.uniform(-0.1,0.1,size=(self.nInput, self.nOutput+1))
        print "Initial weight matrix \n %s \n" % self.W

        for x in range(0, cant_patterns):
            print "Training pattern %d. Input: %s -> Expected output: %s" % (x, dataset[x,0], dataset[x,1])

def main():
    a = PerceptronSimple(2,2)

if __name__ == "__main__":
    main()