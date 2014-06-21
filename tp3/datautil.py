import numpy as np
import matplotlib.pyplot as plt

def getTestSetWithNoise(testset, noiseRate, plot=False):
    cantBits = len(testset)
    cantBitsToSwitch = int(cantBits * noiseRate)

    indicesToSwitch = np.random.permutation(np.arange(cantBits))[:cantBitsToSwitch]

    newset = np.copy(testset)

    for i in indicesToSwitch:
        newset[i] *= -1

    if(plot):
        new_letter = np.copy(newset)
        new_letter = new_letter.reshape(14,14)
        plt.imshow(new_letter, interpolation='none')
        plt.show()

    return newset