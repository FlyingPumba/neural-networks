import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        plt.imshow(new_letter, interpolation='none', cmap=cm.gray)
        plt.show()

    return newset

def plotLetter(letter):
    letterReshaped = np.copy(letter).reshape(14,14)
    plt.imshow(letterReshaped, interpolation='none', cmap=cm.gray)
    plt.show()

def plotLetters(original, output):
    sp1 = plt.subplot(211)
    originalReshaped = np.copy(original).reshape(14,14)
    sp1.imshow(originalReshaped, interpolation='none', cmap=cm.gray)

    sp2 = plt.subplot(212)
    outputReshaped = np.copy(output).reshape(14,14)
    sp2.imshow(outputReshaped, interpolation='none', cmap=cm.gray)

    plt.show()