import numpy as np
import ocrletters as l
import datautil as du

class OCR():
    trainingset = np.asarray([
        l.A, l.B, l.C, l.D, l.E,
        l.F, l.G, l.H, l.I, l.J,
        l.K, l.L, l.M, l.N, l.O,
        l.P, l.Q, l.R, l.S, l.T,
        l.U, l.V, l.W, l.X, l.Y,
        l.Z])
    testset = np.copy(trainingset)