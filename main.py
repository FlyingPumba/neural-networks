import numpy as np
import scipy as sci

class PerceptronSimple():
    """Perceptron Simple"""
    def __init__(self):
        dataset = np.array([[1,2],[1,2]])
        print(dataset.shape)
        self.train_network(dataset)

    def train_network(dataset, batch=False):
        D = np.zeros((2,3))
        #dataset.shape
        #self.D = np.zeros(dataset.shape)
        #print(D.shape)

def main():
    a = PerceptronSimple()

if __name__ == "__main__":
    main()
