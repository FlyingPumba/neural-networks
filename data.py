import numpy as np

class AND():
    trainingset = np.array([
        [[0,0],[0]],
        [[0,1],[0]],
        [[1,0],[0]],
        [[1,1],[1]]
        ])
    testset = np.array([[0,0], [0,1], [1,0], [1,1]])

class OR():
    trainingset = np.array([
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[1]]
        ])
    testset = np.array([[0,0], [0,1], [1,0], [1,1]])

class ANDOR():
    trainingset = np.array([
        [[0,0],[0,0]],
        [[0,1],[1,0]],
        [[1,0],[1,0]],
        [[1,1],[1,1]]
        ])
    testset = np.array([[0,0], [0,1], [1,0], [1,1]])