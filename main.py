import numpy as np
import scipy as sci

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self):
        dataset = np.array([
            [[0,0],[0,0]],
            [[0,1],[1,0]],
            [[1,0],[1,0]],
            [[1,1],[1,1]]
            ])
        self.train_network(dataset)

    def train_network(self, dataset, batch=False):
        print "Data set size is:"
        print dataset.shape
        print "\n"

        # how many training patterns do we have ?
        cant_patterns = dataset.shape[0]

        # create D matrix for batch learning
        D = np.zeros(dataset.shape)

        for x in range(0, cant_patterns):
            print "Training pattern %d. Input: %s -> Expected output: %s" % (x, dataset[x,0], dataset[x,1])

def main():
    a = PerceptronSimple()

if __name__ == "__main__":
    main()