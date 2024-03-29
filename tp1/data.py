import numpy as np
import ocrletters as l
import datautil as du

booleanInput = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])

class AND():
    trainingset = np.array([
        [[-1,-1],[-1]],
        [[-1,1],[-1]],
        [[1,-1],[-1]],
        [[1,1],[1]]
        ])
    testset = trainingset

class OR():
    trainingset = np.array([
        [[-1,-1],[-1]],
        [[-1,1],[1]],
        [[1,-1],[1]],
        [[1,1],[1]]
        ])
    testset = trainingset

class ANDOR():
    trainingset = np.array([
        [[-1,-1],[-1,-1]],
        [[-1,1],[1,-1]],
        [[1,-1],[1,-1]],
        [[1,1],[1,1]]
        ])
    testset = trainingset

class XOR():
    trainingset = np.array([
        [[-1,-1],[-1]],
        [[-1,1],[1]],
        [[1,-1],[1]],
        [[1,1],[-1]]
        ])
    testset = trainingset

class OCR():
    trainingset = np.asarray([
        [l.A, np.array([0,0,0,0,1])], # A -> 1
        [l.B, np.array([0,0,0,1,0])], # B -> 2
        [l.C, np.array([0,0,0,1,1])], # C -> 3
        [l.D, np.array([0,0,1,0,0])], # D -> 4
        [l.E, np.array([0,0,1,0,1])], # E -> 5
        [l.F, np.array([0,0,1,1,0])], # F -> 6
        [l.G, np.array([0,0,1,1,1])], # G -> 7
        [l.H, np.array([0,1,0,0,0])], # H -> 8
        [l.I, np.array([0,1,0,0,1])], # I -> 9
        [l.J, np.array([0,1,0,1,0])], # J -> 10
        [l.K, np.array([0,1,0,1,1])], # K -> 11
        [l.L, np.array([0,1,1,0,0])], # L -> 12
        [l.M, np.array([0,1,1,0,1])], # M -> 13
        [l.N, np.array([0,1,1,1,0])], # N -> 14
        [l.O, np.array([0,1,1,1,1])], # O -> 15
        [l.P, np.array([1,0,0,0,0])], # P -> 16
        [l.Q, np.array([1,0,0,0,1])], # Q -> 17
        [l.R, np.array([1,0,0,1,0])], # R -> 18
        [l.S, np.array([1,0,0,1,1])], # S -> 19
        [l.T, np.array([1,0,1,0,0])], # T -> 20
        [l.U, np.array([1,0,1,0,1])], # U -> 21
        [l.V, np.array([1,0,1,1,0])], # V -> 22
        [l.W, np.array([1,0,1,1,1])], # W -> 23
        [l.X, np.array([1,1,0,0,0])], # X -> 24
        [l.Y, np.array([1,1,0,0,1])], # Y -> 25
        [l.Z, np.array([1,1,0,1,0])], # Z -> 26
        ])
    testset = trainingset

class BinaryOCR(OCR):
    # Idem to OCR
    trainingset = OCR.trainingset
    testset = trainingset

class BipolarOCR(OCR):
    # Swithc 0s for -1s
    trainingset = np.copy(OCR.trainingset)
    cant_patterns = trainingset.shape[0]
    for i in range(0, cant_patterns):
        trainingset[i,0] = OCR.trainingset[i,0]*2-1
        trainingset[i,1] = OCR.trainingset[i,1]*2-1
    testset = trainingset

class Sin():
    trainingset = []
    interpolationset = []
    extrapolationset = []

    flag = True
    flag2 = 0
    x = 0
    y = 0
    for val in du.rangef(0,2*np.pi,0.05):
        if(flag):
            x = np.random.uniform(-10,10)
            y = 2*(val - x)
            flag = False
        else:
            y = np.random.uniform(-10,10)
            x = val - y/2
            flag = True

        if flag2 < 2:
            trainingset.append([[x,y],[np.sin(val)]])
            flag2 = flag2 + 1
        else:
            interpolationset.append([[x,y],[np.sin(val)]])
            flag2 = 0

    trainingset = np.asarray(trainingset)
    interpolationset = np.asarray(interpolationset)

    #extrapolation set: [-2pi,0] U [2pi,4pi]
    for val in du.rangef(-2*np.pi,0,0.1):
        if(flag):
            x = np.random.uniform(-10,10)
            y = 2*(val - x)
            flag = False
        else:
            y = np.random.uniform(-10,10)
            x = val - y/2
            flag = True
        extrapolationset.append([[x,y],[np.sin(val)]])

    for val in du.rangef(2*np.pi,4*np.pi,0.1):
        if(flag):
            x = np.random.uniform(-10,10)
            y = 2*(val - x)
            flag = False
        else:
            y = np.random.uniform(-10,10)
            x = val - y/2
            flag = True
        extrapolationset.append([[x,y],[np.sin(val)]])

    extrapolationset = np.asarray(extrapolationset)