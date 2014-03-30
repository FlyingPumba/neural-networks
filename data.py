import numpy as np
import ocrletters as l

booleanInput = np.array([[0,0], [0,1], [1,0], [1,1]])

class AND():
    trainingset = np.array([
        [[0,0],[0]],
        [[0,1],[0]],
        [[1,0],[0]],
        [[1,1],[1]]
        ])
    testset = booleanInput

class OR():
    trainingset = np.array([
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[1]]
        ])
    testset = booleanInput

class ANDOR():
    trainingset = np.array([
        [[0,0],[0,0]],
        [[0,1],[1,0]],
        [[1,0],[1,0]],
        [[1,1],[1,1]]
        ])
    testset = booleanInput

class XOR():
    trainingset = np.array([
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[0]]
        ])
    testset = booleanInput

class SimpleOCR():
    trainingset = np.asarray([
        [l.A, [0,0,0,0,1]], # A -> 1
        [l.B, [0,0,0,1,0]], # B -> 2
        [l.C, [0,0,0,1,1]], # C -> 3
        [l.D, [0,0,1,0,0]], # D -> 4
        [l.E, [0,0,1,0,1]], # E -> 5
        [l.F, [0,0,1,1,0]], # F -> 6
        [l.G, [0,0,1,1,1]], # G -> 7
        [l.H, [0,1,0,0,0]], # H -> 8
        [l.I, [0,1,0,0,1]], # I -> 9
        [l.J, [0,1,0,1,0]], # J -> 10
        [l.K, [0,1,0,1,1]], # K -> 11
        [l.L, [0,1,1,0,0]], # L -> 12
        [l.M, [0,1,1,0,1]], # M -> 13
        [l.N, [0,1,1,1,0]], # N -> 14
        [l.O, [0,1,1,1,1]], # O -> 15
        [l.P, [1,0,0,0,0]], # P -> 16
        [l.Q, [1,0,0,0,1]], # Q -> 17
        [l.R, [1,0,0,1,0]], # R -> 18
        [l.S, [1,0,0,1,1]], # S -> 19
        [l.T, [1,0,1,0,0]], # T -> 20
        [l.U, [1,0,1,0,1]], # U -> 21
        [l.V, [1,0,1,1,0]], # V -> 22
        [l.W, [1,0,1,1,1]], # W -> 23
        [l.X, [1,1,0,0,0]], # X -> 24
        [l.Y, [1,1,0,0,1]], # Y -> 25
        [l.Z, [1,1,0,1,0]], # Z -> 26
        ])