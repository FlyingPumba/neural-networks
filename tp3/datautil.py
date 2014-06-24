import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def getPatternWithNoise(pattern, noiseRate, plot=False):
    cantBits = len(pattern)
    cantBitsToSwitch = int(cantBits * noiseRate)

    indicesToSwitch = np.random.permutation(np.arange(cantBits))[:cantBitsToSwitch]

    newpattern = np.copy(pattern)

    for i in indicesToSwitch:
        newpattern[i] *= -1

    if(plot):
        # assuming it is an OCR letter
        new_letter = np.copy(newpattern)
        new_letter = new_letter.reshape(14,14)
        plt.imshow(new_letter, interpolation='none', cmap=cm.gray)
        plt.show()

    return newpattern

def plotLetter(letter, saveFile=False):
    letterReshaped = np.copy(letter).reshape(14,14)
    plt.imshow(letterReshaped, interpolation='none', cmap=cm.gray)
    if saveFile:
        filekey = np.random.randint(1000)
        fileName = 'hopfield1-energy-%d.png' % filekey
        print fileName
        figure = plt.gcf() # get current figure
        figure.set_size_inches(10, 8) #this will give us a 800x600 image
        # when saving, specify the DPI
        plt.savefig(fileName, bbox_inches='tight', dpi = 100)
    plt.show()

def plotLetters(original, output):
    sp1 = plt.subplot(211)
    originalReshaped = np.copy(original).reshape(14,14)
    sp1.imshow(originalReshaped, interpolation='none', cmap=cm.gray)

    sp2 = plt.subplot(212)
    outputReshaped = np.copy(output).reshape(14,14)
    sp2.imshow(outputReshaped, interpolation='none', cmap=cm.gray)

    plt.show()