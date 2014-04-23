import numpy as np
import matplotlib.pyplot as plt

def getTestSetWithNoise(testset, noiseRate, plot=False):
    cant_patterns = testset.shape[0]
    new_testset = []
    for i in range(0, cant_patterns):
        # alter the pattern
        new_testset.append([alterPattern(noiseRate, testset[i,0]), testset[i,1]])
        #print "original \n %s" % testset[i]
        #print "with noise \n %s" % new_testset[i]

        if(plot):
            new_letter = new_testset[i][0]
            new_letter = new_letter.reshape(5,5)
            print "letter reshaped \n %s" % new_letter
            plt.imshow(new_letter, interpolation='none')
            plt.title("letter %d" % i)
            plt.show()
    #raw_input()
    return np.array(new_testset)

def alterPattern(noiseRate, pattern):
    return pattern + np.random.uniform(-noiseRate/2,noiseRate/2, size=pattern.shape)

def getTestSetWithSwitchedUnits(testset, noiseRate, plot=False):
    cant_patterns = testset.shape[0]
    new_testset = []
    for i in range(0, cant_patterns):
        # alter the pattern
        A = np.random.uniform(0, 1, testset[i,0].shape)
        B = np.where(A<noiseRate, -1, 1)
        new_testset.append(np.array([np.array(testset[i,0] * B), testset[i,1]]))
        #print "original \n %s" % testset[i]
        #print "with switched units \n %s" % new_testset[i]

        if(plot):
            new_letter = new_testset[i][0]
            new_letter = new_letter.reshape(5,5)
            print "letter reshaped \n %s" % new_letter
            plt.imshow(new_letter, interpolation='none')
            plt.title("letter %d" % i)
            plt.show()
    #raw_input()
    return np.array(new_testset)

def rangef(min, max, step):
    while min<max:
        yield min
        min = min+step